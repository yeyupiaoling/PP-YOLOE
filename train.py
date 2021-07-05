import ppdet.utils.check as check
import ppdet.utils.cli as cli
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger

logger = setup_logger('train')


def parse_args():
    parser = cli.ArgsParser()
    parser.add_argument("--config",
                        type=str,
                        default="config_ppyolo_tiny/ppyolo_tiny_650e_voc.yml",
                        choices=['config_ppyolo_tiny/ppyolo_tiny_650e_voc.yml', 'config_ppyolo/ppyolo_r50vd_dcn_1x_voc.yml'],
                        help="所使用的模型，有PPYOLO和PPYOLO tiny选择。")
    parser.add_argument("--eval",
                        action='store_true',
                        default=True,
                        help="是否在训练过程中执行评估。")
    parser.add_argument("--resume",
                        default=None,
                        help="恢复之前训练的状态，路径不能包含模型后缀名。")
    parser.add_argument("--slim_config",
                        type=str,
                        default='config_ppyolo_tiny/ppyolo_mbv3_large_qat.yml',
                        choices=['config_ppyolo_tiny/ppyolo_mbv3_large_qat.yml', 'config_ppyolo/ppyolo_r50vd_qat_pact.yml'],
                        help="使用量化训练的配置文件路径，设置为None则不使用量化训练。")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # init parallel environment if nranks > 1
    init_parallel_env()

    # build trainer
    trainer = Trainer(cfg, mode='train')

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    for key in cfg.keys():
        value = cfg[key]
        if value != {}:
            print(key, ":", value)
    cfg['fp16'] = False
    cfg['fleet'] = False
    cfg['use_vdl'] = True
    cfg['vdl_log_dir'] = "log"
    cfg['save_prediction_only'] = False
    merge_config(FLAGS.opt)

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config)

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()
