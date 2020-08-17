# PP-YOLO

PP-YOLO是PaddleDetection优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型，PP-YOLO在COCO test-dev2017数据集上精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。


## 训练

1. 安装PaddlePaddle GPU版本，需要1.8.4以上。
```shell script
python -m pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple
```

2. 安装ppdet，以及所需依赖包。
```shell script
python setup.py
```

3. 准备数据，默认使用的是VOC格式的数据集，如果要修改为COCO格式的数据集，需要修改`configs/ppyolo.yml`。其中VOC格式操作如下，首先将标注文件放在`dataset/annotation/`，图像文件放在`dataset/images/`，最后执行`create_data_list.py`程序生成数据列表和类别名称。

4. 修改`configs/ppyolo.yml`中的类别数量`num_classes`，这个类别数量不用算上背景这一类别。

5. 执行`train.py`开始训练。


## 评估和预测