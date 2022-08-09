# PP-YOLOE

PP-YOLOE是单阶段Anchor-free模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv5模型，PP-YOLOE在COCO test-dev2017数据集上精度达到49.0%，在单卡V100上FP32推理速度为123.4FPS, V100上开启TensorRT下FP16推理速度为208.3FPS。其中还包含了X/L/M/S四种模型，适合部署在多种多样的硬件上，在手机上部署，推理速度也是极快的。

# 模型表

| 模型类型 | Epoch | 输入尺寸 | Box AP<sup>val<br>0.5:0.95 | Box AP<sup>test<br>0.5:0.95 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |
|:----:|:-----:|:----:|:--------------------------:|:---------------------------:|:---------:|:--------:|:--------------:|:-----------------------:|
|  S   |  300  | 640  |            43.0            |            43.2             |   7.93    |  17.36   |     208.3      |          333.3          |
|  M   |  300  | 640  |            49.0            |            49.1             |   23.43   |  49.91   |     123.4      |          208.3          |
|  L   |  300  | 640  |            51.4            |            51.6             |   52.20   |  110.07  |      78.1      |          149.2          | 
|  X   |  300  | 640  |            52.3            |            52.4             |   98.42   |  206.59  |      45.0      |          95.2           | 


# 安装环境
1. 安装PaddlePaddle GPU版本
```shell
conda install paddlepaddle-gpu==2.3.1 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

2. 其他依赖库
```shell
python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
