import distutils.util
import os
import shutil

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


def get_coco_model(model_type: str, pretrained_dir='pretrained_models'):
    assert model_type.upper() in ["X", "L", "M", "S"]
    url = f"https://paddledet.bj.bcebos.com/models/ppyoloe_crn_{model_type.lower()}_300e_coco.pdparams"
    pretrained_model_path = os.path.join(pretrained_dir, f"ppyoloe_crn_{model_type.lower()}_300e_coco.pdparams")
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
