import argparse
import functools
import os
import shutil
import time
from datetime import timedelta

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from visualdl import LogWriter

from data_utils.reader import CustomDataset, BatchCompose
from metrics.metrics import COCOMetric
from model.yolo import PPYOLOE_S, PPYOLOE_M, PPYOLOE_L, PPYOLOE_X
from utils.logger import setup_logger
from utils.lr import cosine_decay_with_warmup
from utils.utils import add_arguments, print_arguments
from utils.utils import get_pretrained_model

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',          str,    'M',                      '所使用的模型类型', choices=["X", "L", "M", "S"])
add_arg('batch_size',          int,    8,                        '训练的批量大小')
add_arg('num_workers',         int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',           int,    80,                       '训练的轮数')
add_arg('num_classes',         int,    80,                       '分类的类别数量')
add_arg('learning_rate',       float,  0.000125,                 '初始学习率的大小')
add_arg('image_dir',           str,    'dataset/',               '图片存放的路径')
add_arg('train_anno_path',     str,    'dataset/train.json',     '训练数据标注信息json文件路径')
add_arg('eval_anno_path',      str,    'dataset/eval.json',      '评估标注信息json文件路径')
add_arg('save_model_dir',      str,    'output/',                '模型保存的路径')
add_arg('use_random_distort',  bool,   True,                     '是否使用随机颜色失真数据增强')
add_arg('use_random_expand',   bool,   True,                     '是否使用随机扩张数据增强')
add_arg('use_random_crop',     bool,   True,                     '是否使用随机裁剪数据增强')
add_arg('use_random_flip',     bool,   True,                     '是否使用随机翻转数据增强')
add_arg('resume_model',        str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
add_arg('pretrained_model',    str,    None,                     '预训练模型的模型文件，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args)


# 评估模型
@paddle.no_grad()
def evaluate(model, eval_loader, metrics: COCOMetric):
    model.eval()
    for batch_id, data in enumerate(eval_loader()):
        outputs = model(data)
        metrics.update(inputs=data, outputs=outputs)
    result = metrics.accumulate()
    metrics.reset()
    model.train()
    return result


# 保存模型
def save_model(save_model_dir, use_model, epoch, model, optimizer, best_model=False):
    if not best_model:
        model_path = os.path.join(save_model_dir, use_model, 'epoch_{}'.format(epoch))
        os.makedirs(model_path, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
        last_model_path = os.path.join(save_model_dir, use_model, 'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)
        # 删除旧的模型
        old_model_path = os.path.join(save_model_dir, use_model, 'epoch_{}'.format(epoch - 3))
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)
    else:
        model_path = os.path.join(save_model_dir, use_model, 'best_model')
        paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    logger.info('已保存模型：{}'.format(model_path))


# 训练模型
def train():
    # 获取有多少张显卡训练
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        # 初始化Fleet环境
        fleet.init(is_collective=True)
    if local_rank == 0:
        # 日志记录器
        writer = LogWriter(logdir='log')
    # 获取数据
    train_dataset = CustomDataset(image_dir=args.image_dir,
                                  anno_path=args.train_anno_path,
                                  data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
                                  mode='train',
                                  use_random_distort=args.use_random_distort,
                                  use_random_expand=args.use_random_expand,
                                  use_random_crop=args.use_random_crop,
                                  use_random_flip=args.use_random_flip)
    train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                            drop_last=True)
    collate_fn = BatchCompose()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_sampler=train_batch_sampler,
                              collate_fn=collate_fn,
                              num_workers=args.num_workers)
    # 评估数据
    eval_dataset = CustomDataset(image_dir=args.image_dir,
                                 anno_path=args.eval_anno_path,
                                 data_fields=['image'],
                                 mode='eval')
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # 获取模型
    if args.model_type == 'X':
        model = PPYOLOE_X(num_classes=args.num_classes)
    elif args.model_type == 'L':
        model = PPYOLOE_L(num_classes=args.num_classes)
    elif args.model_type == 'M':
        model = PPYOLOE_M(num_classes=args.num_classes)
    elif args.model_type == 'S':
        model = PPYOLOE_S(num_classes=args.num_classes)
    else:
        raise Exception(f'模型类型不存在，model_type：{args.model_type}')

    if nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 评估方法
    metrics = COCOMetric(anno_file=args.eval_anno_path)

    # 学习率衰减
    scheduler = cosine_decay_with_warmup(learning_rate=args.learning_rate * nranks,
                                         max_epochs=int(args.num_epoch*1.2),
                                         step_per_epoch=len(train_loader))
    # 设置优化方法
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=scheduler,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(5e-4))

    if nranks > 1:
        # 设置支持多卡训练
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    # 加载预训练模型
    if args.pretrained_model is not None:
        assert os.path.exists(args.pretrained_model), f"预训练模型不存在，路径：{args.pretrained_model}"
        model_dict = model.state_dict()
        model_state_dict = paddle.load(args.pretrained_model)
        for name, weight in model_dict.items():
            if name in model_state_dict.keys():
                if weight.shape != list(model_state_dict[name].shape):
                    print('{} not used, shape {} unmatched with {} in model.'.
                          format(name, list(model_state_dict[name].shape), weight.shape))
                    model_state_dict.pop(name, None)
            else:
                print('Lack weight: {}'.format(name))
        model.set_state_dict(model_state_dict)
        logger.info('成功加载预训练模型：{}'.format(args.pretrained_model))
    else:
        # 加载官方预训练模型
        pretrained_path = get_pretrained_model(model_type=args.model_type)
        model.set_state_dict(paddle.load(pretrained_path))
        logger.info('成功加载预训练模型：{}'.format(pretrained_path))

    # 加载恢复模型
    last_epoch = 0
    if args.resume_model is not None:
        assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(args.resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
        optimizer_state = paddle.load(os.path.join(args.resume_model, 'optimizer.pdopt'))
        optimizer.set_state_dict(optimizer_state)
        last_epoch = optimizer_state['LR_Scheduler']['last_epoch'] // len(train_loader)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(args.resume_model))

    train_times = []
    best_mAP, train_step, test_step = 0, 0, 0
    sum_batch = len(train_loader) * args.num_epoch
    # 开始训练
    for epoch_id in range(last_epoch, args.num_epoch):
        start_epoch = time.time()
        start = time.time()
        loss_sum = []
        for batch_id, data in enumerate(train_loader()):
            data['epoch_id'] = epoch_id
            output = model(data)
            loss = output['loss']
            loss_sum.append(loss.numpy()[0])
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_times.append((time.time() - start) * 1000)
            # 打印
            if batch_id % 100 == 0:
                eta_sec = (sum(train_times) / len(train_times)) * (sum_batch - epoch_id * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch [{epoch_id}/{args.num_epoch}], '
                            f'batch: [{batch_id}/{len(train_loader)}], '
                            f'loss: {(sum(loss_sum) / len(loss_sum)):.5f}, '
                            f'lr: {scheduler.get_lr():.8f}, '
                            f'eta: {eta_str}')
                train_times = []
            if local_rank == 0:
                writer.add_scalar('Train/Learning rate', scheduler.get_lr(), train_step)
                writer.add_scalar('Train/Loss', (sum(loss_sum) / len(loss_sum)), train_step)
                train_step += 1
            start = time.time()
            scheduler.step()
        if local_rank == 0:
            # 保存模型
            save_model(save_model_dir=args.save_model_dir, use_model=f'PPYOLOE_{args.model_type.upper()}',
                       epoch=epoch_id, model=model, optimizer=optimizer)
        print('\n', '=' * 70)
        # 执行评估
        mAP = evaluate(model=model, eval_loader=eval_loader, metrics=metrics)[0]
        if local_rank == 0:
            writer.add_scalar('Test/mAP', mAP, test_step)
            test_step += 1
        if mAP >= best_mAP:
            best_mAP = mAP
            if local_rank == 0:
                # 保存效果最好的模型
                save_model(save_model_dir=args.save_model_dir, use_model=f'PPYOLOE_{args.model_type.upper()}',
                           epoch=epoch_id, model=model, optimizer=optimizer, best_model=True)
        logger.info('Test epoch: {}, time/epoch: {}, mAP: {:.5f}, best_mAP: {:.5f}'.format(
            epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), mAP, best_mAP))
        print('=' * 70, '\n')


if __name__ == '__main__':
    train()
