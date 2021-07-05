import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer, init_parallel_env
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger

logger = setup_logger('eval')


def parse_args():
    parser = ArgsParser()
    parser.add_argument("--config",
                        type=str,
                        default="config_ppyolo_tiny/ppyolo_tiny_650e_voc.yml",
                        choices=['config_ppyolo_tiny/ppyolo_tiny_650e_voc.yml', 'config_ppyolo/ppyolo_r50vd_dcn_1x_voc.yml'],
                        help="所使用的模型，有PPYOLO和PPYOLO tiny选择。")
    parser.add_argument("--slim_config",
                        type=str,
                        default='config_ppyolo_tiny/ppyolo_mbv3_large_qat.yml',
                        choices=['config_ppyolo_tiny/ppyolo_mbv3_large_qat.yml', 'config_ppyolo/ppyolo_r50vd_qat_pact.yml'],
                        help="使用量化训练的配置文件路径，设置为None则不使用量化训练。")
    args = parser.parse_args()
    return args


def run(cfg):
    # init parallel environment if nranks > 1
    init_parallel_env()

    # build trainer
    trainer = Trainer(cfg, mode='eval')

    # load weights
    trainer.load_weights(cfg.weights)

    # training
    trainer.evaluate()


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['bias'] = 0
    cfg['classwise'] = False
    cfg['output_eval'] = None
    cfg['save_prediction_only'] = False
    merge_config(FLAGS.opt)

    paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='eval')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(cfg)


if __name__ == '__main__':
    main()
