import distutils.util
import os
import shutil

import paddle
import requests
from tqdm import tqdm

from utils.logger import setup_logger

logger = setup_logger(__name__)


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (paddle.arange(end=w) + grid_cell_offset) * stride
        shift_y = (paddle.arange(end=h) + grid_cell_offset) * stride
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        anchor = paddle.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            axis=-1).astype(feat.dtype)
        anchor_point = paddle.stack(
            [shift_x, shift_y], axis=-1).astype(feat.dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            paddle.full(
                [num_anchors_list[-1], 1], stride, dtype=feat.dtype))
    anchors = paddle.concat(anchors)
    anchors.stop_gradient = True
    anchor_points = paddle.concat(anchor_points)
    anchor_points.stop_gradient = True
    stride_tensor = paddle.concat(stride_tensor)
    stride_tensor.stop_gradient = True
    return anchors, anchor_points, num_anchors_list, stride_tensor


def get_pretrained_model(model_type: str, pretrained_dir='pretrained_models'):
    assert model_type.upper() in ["X", "L", "M", "S"]
    url = f"https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_{model_type.lower()}_pretrained.pdparams"
    pretrained_model_path = os.path.join(pretrained_dir, f"CSPResNetb_{model_type.lower()}_pretrained.pdparams")
    if os.path.exists(pretrained_model_path):
        return pretrained_model_path
    else:
        logger.info("开始下载预训练模型")
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code {}!".format(url, req.status_code))

        # 避免下载中断，先保存为临时文件
        tmp_fullname = pretrained_model_path + "_tmp"
        total_size = req.headers.get('content-length')
        os.makedirs(os.path.dirname(tmp_fullname), exist_ok=True)
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, pretrained_model_path)
        logger.info(f"预训练模型下载成功，保存路径：{pretrained_model_path}")
        return pretrained_model_path
