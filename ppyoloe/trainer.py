import json
import json
import os
import shutil
import time
from datetime import timedelta

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from tqdm import tqdm
from visualdl import LogWriter

from ppyoloe import SUPPORT_MODEL
from ppyoloe.data_utils.reader import CustomDataset, BatchCompose
from ppyoloe.metrics.metrics import COCOMetric
from ppyoloe.model.utils import get_infer_cfg_and_input_spec
from ppyoloe.model.yolo import PPYOLOE_S, PPYOLOE_M, PPYOLOE_L, PPYOLOE_X
from ppyoloe.utils.logger import setup_logger
from ppyoloe.utils.lr import cosine_decay_with_warmup
from ppyoloe.utils.utils import get_coco_model
from ppyoloe.utils.utils import get_pretrained_model

logger = setup_logger(__name__)


class PPYOLOETrainer(object):
    def __init__(self,
                 model_type='M',
                 batch_size=8,
                 num_workers=8,
                 num_classes=80,
                 image_dir='dataset/',
                 train_anno_path='dataset/train.json',
                 eval_anno_path='dataset/eval.json',
                 use_gpu=True):
        """PPYOLOE集成工具类

        :param model_type: 所使用的模型类型
        :param batch_size: 训练或者评估的批量大小
        :param num_workers: 读取数据的线程数量
        :param num_classes: 分类的类别数量
        :param image_dir: 图片存放的路径
        :param train_anno_path: 训练数据标注信息json文件路径
        :param eval_anno_path: 评估标注信息json文件路径
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        self.use_gpu = use_gpu
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.train_anno_path = train_anno_path
        self.eval_anno_path = eval_anno_path
        assert self.model_type in SUPPORT_MODEL, f'没有该模型：{self.model_type}'
        self.model = None
        self.test_loader = None
        self.metrics = None

    def __setup_dataloader(self,
                           use_random_distort=True,
                           use_random_expand=True,
                           use_random_crop=True,
                           use_random_flip=True,
                           eval_image_size=[640, 640],
                           is_train=False):
        if is_train:
            # 获取数据
            self.train_dataset = CustomDataset(image_dir=self.image_dir,
                                               anno_path=self.train_anno_path,
                                               data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
                                               mode='train',
                                               use_random_distort=use_random_distort,
                                               use_random_expand=use_random_expand,
                                               use_random_crop=use_random_crop,
                                               use_random_flip=use_random_flip)
            train_batch_sampler = paddle.io.DistributedBatchSampler(self.train_dataset, batch_size=self.batch_size,
                                                                    shuffle=True, drop_last=True)
            collate_fn = BatchCompose()
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_sampler=train_batch_sampler,
                                           collate_fn=collate_fn,
                                           num_workers=self.num_workers)
        # 评估数据
        test_dataset = CustomDataset(image_dir=self.image_dir,
                                     anno_path=self.eval_anno_path,
                                     eval_image_size=eval_image_size,
                                     data_fields=['image'],
                                     mode='eval')
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)

    def __setup_model(self, num_epoch=80, learning_rate=1.25e-4, is_train=False):
        # 获取模型
        if self.model_type == 'X':
            self.model = PPYOLOE_X(num_classes=self.num_classes)
        elif self.model_type == 'L':
            self.model = PPYOLOE_L(num_classes=self.num_classes)
        elif self.model_type == 'M':
            self.model = PPYOLOE_M(num_classes=self.num_classes)
        elif self.model_type == 'S':
            self.model = PPYOLOE_S(num_classes=self.num_classes)
        else:
            raise Exception(f'模型类型不存在，model_type：{self.model_type}')
        # print(self.model)
        if paddle.distributed.get_world_size() > 1:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if is_train:
            # 学习率衰减
            self.scheduler = cosine_decay_with_warmup(learning_rate=learning_rate * paddle.distributed.get_world_size(),
                                                      max_epochs=int(num_epoch * 1.2),
                                                      step_per_epoch=len(self.train_loader))
            # 设置优化方法
            self.optimizer = paddle.optimizer.Momentum(parameters=self.model.parameters(),
                                                       learning_rate=self.scheduler,
                                                       momentum=0.9,
                                                       weight_decay=paddle.regularizer.L2Decay(5e-4))

    def __load_pretrained(self, pretrained_model=None):
        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            model_dict = self.model.state_dict()
            model_state_dict = paddle.load(pretrained_model)
            # 过滤不存在的参数
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            self.model.set_state_dict(model_state_dict)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))
        else:
            # 加载官方预训练模型
            pretrained_path = get_pretrained_model(model_type=self.model_type)
            self.model.set_state_dict(paddle.load(pretrained_path))
            logger.info('成功加载预训练模型：{}'.format(pretrained_path))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_mAP = 0
        last_model_dir = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
            self.model.set_state_dict(paddle.load(os.path.join(resume_model, 'model.pdparams')))
            self.optimizer.set_state_dict(paddle.load(os.path.join(resume_model, 'optimizer.pdopt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_mAP = json_data['mAP']
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        return last_epoch, best_mAP

    # 保存模型
    def __save_checkpoint(self, save_model_path, epoch_id, mAP=0, best_model=False):
        if best_model:
            model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'best_model')
        else:
            model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        try:
            paddle.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            paddle.save(self.model.state_dict(), os.path.join(model_path, 'model.pdparams'))
        except Exception as e:
            logger.error(f'保存模型时出现错误，错误信息：{e}')
            return
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            f.write('{"last_epoch": %d, "mAP": %f}' % (epoch_id, mAP))
        if not best_model:
            last_model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}', 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, f'PPYOLOE_{self.model_type}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, max_epoch, epoch_id, log_interval, local_rank, writer):
        train_times, loss_sum = [], []
        start = time.time()
        sum_batch = len(self.train_loader) * max_epoch
        for batch_id, data in enumerate(self.train_loader()):
            data['epoch_id'] = epoch_id
            output = self.model(data)
            # 计算损失值
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            loss_sum.append(loss.numpy()[0])
            train_times.append((time.time() - start) * 1000)

            # 多卡训练只使用一个进程打印
            if batch_id % log_interval == 0 and local_rank == 0:
                # 计算每秒训练数据量
                train_speed = self.batch_size / (sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'learning rate: {self.scheduler.get_lr():>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                train_times = []
            self.scheduler.step()
            # 记录学习率
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), self.train_step)
            start = time.time()

    def train(self,
              num_epoch=80,
              learning_rate=1.25e-4,
              log_interval=100,
              use_random_distort=True,
              use_random_expand=True,
              use_random_crop=True,
              use_random_flip=True,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param num_epoch: 训练的轮数
        :param learning_rate: 初始学习率的大小
        :param log_interval: 指定步数打印一次日志
        :param use_random_distort: 是否使用随机颜色失真数据增强
        :param use_random_expand: 是否使用随机扩张数据增强
        :param use_random_crop: 是否使用随机裁剪数据增强
        :param use_random_flip: 是否使用随机翻转数据增强
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        paddle.seed(1000)
        # 获取有多少张显卡训练
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化Fleet环境
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)

        # 获取数据
        self.__setup_dataloader(use_random_distort=use_random_distort,
                                use_random_expand=use_random_expand,
                                use_random_crop=use_random_crop,
                                use_random_flip=use_random_flip,
                                is_train=True)
        # 获取模型
        self.__setup_model(num_epoch=num_epoch, learning_rate=learning_rate, is_train=True)
        # 评估方法
        self.metrics = COCOMetric(anno_file=self.eval_anno_path)
        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_mAP = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), last_epoch)
        # 开始训练
        for epoch_id in range(last_epoch, num_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(max_epoch=num_epoch, epoch_id=epoch_id, log_interval=log_interval, local_rank=local_rank, writer=writer)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0:
                logger.info('=' * 70)
                mAP = self.evaluate(resume_model=None)[0]
                # 保存最优模型
                if mAP >= best_mAP:
                    best_mAP = mAP
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, mAP=mAP,
                                           best_model=True)
                logger.info('Test epoch: {}, time/epoch: {}, best_mAP: {:.5f}, mAP: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), best_mAP, mAP))
                logger.info('=' * 70)
                writer.add_scalar('Test/mAP', mAP, test_step)
                test_step += 1
                self.model.train()
                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, mAP=mAP)

    def evaluate(self, image_size='640,640', resume_model='models/PPYOLOE_M/best_model/'):
        """
        评估模型
        :param image_size: 评估时图像输入大小
        :param resume_model: 所使用的模型
        :return: 评估结果

        """
        if self.metrics is None:
            self.metrics = COCOMetric(anno_file=self.eval_anno_path)
        if self.test_loader is None:
            eval_image_size = [int(s) for s in image_size.split(',')]
            self.__setup_dataloader(eval_image_size=eval_image_size)
        if self.model is None:
            self.__setup_model()
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = paddle.load(resume_model)
            self.model.set_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, paddle.DataParallel):
            eval_model = self.model._layers
        else:
            eval_model = self.model

        with paddle.no_grad():
            for batch_id, data in enumerate(tqdm(self.test_loader())):
                outputs = eval_model(data)
                self.metrics.update(inputs=data, outputs=outputs)
        mAP = self.metrics.accumulate()
        self.metrics.reset()
        self.model.train()
        return mAP

    def export(self, image_shape='3,640,640', save_model_path='models/', resume_model='models/PPYOLOE_M/best_model/'):
        """
        导出预测模型
        :param image_shape: 导出模型图像输入大小
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取模型
        self.__setup_model()
        # 加载预训练模型
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = paddle.load(resume_model)
            self.model.set_state_dict(model_state_dict)
            logger.info('成功恢复模型参数数：{}'.format(resume_model))
        else:
            # 加载官方预训练模型
            pretrained_path = get_coco_model(model_type=self.model_type)
            self.model.set_state_dict(paddle.load(pretrained_path))
            logger.info('成功加载预训练模型：{}'.format(pretrained_path))
        self.model.eval()
        # 获取静态模型
        image_shape = [int(i) for i in image_shape.split(',')]
        static_model, pruned_input_spec = get_infer_cfg_and_input_spec(model=self.model, image_shape=image_shape)
        infer_model_dir = os.path.join(save_model_path, f'PPYOLOE_{self.model_type.upper()}', 'infer')
        paddle.jit.save(static_model, os.path.join(infer_model_dir, 'model'), input_spec=pruned_input_spec)
        logger.info(f'导出模型保存在：{infer_model_dir}')
