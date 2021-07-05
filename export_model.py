import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.slim import build_slim_model
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.logger import setup_logger

logger = setup_logger('export_model')


def parse_args():
    parser = ArgsParser()
    parser.add_argument("--config",
                        type=str,
                        default="config_ppyolo_tiny/ppyolo_tiny_650e_voc.yml",
                        choices=['config_ppyolo_tiny/ppyolo_tiny_650e_voc.yml', 'config_ppyolo/ppyolo_r50vd_dcn_1x_voc.yml'],
                        help="所使用的模型，有PPYOLO和PPYOLO tiny选择。")
    parser.add_argument("--output_dir",
                        type=str,
                        default="output_inference",
                        help="导出预测模型的路径。")
    parser.add_argument("--slim_config",
                        type=str,
                        default='config_ppyolo_tiny/ppyolo_mbv3_large_qat.yml',
                        choices=['config_ppyolo_tiny/ppyolo_mbv3_large_qat.yml', 'config_ppyolo/ppyolo_r50vd_qat_pact.yml'],
                        help="使用量化训练的配置文件路径，设置为None则不使用量化训练。")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build detector
    trainer = Trainer(cfg, mode='test')

    # load weights
    trainer.load_weights(cfg.weights)

    # export model
    trainer.export(FLAGS.output_dir)


def main():
    paddle.set_device("cpu")
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn':
        FLAGS.opt['norm_type'] = 'bn'
    merge_config(FLAGS.opt)

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
