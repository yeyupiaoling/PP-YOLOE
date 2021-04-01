# PP-YOLO

PP-YOLO是PaddleDetection优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型，PP-YOLO在COCO test-dev2017数据集上精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。


## 训练

1. 安装PaddlePaddle GPU版本，需要2.0.1。
```shell script
python -m pip install paddlepaddle-gpu==2.0.1.post101 -i https://mirror.baidu.com/pypi/simple
```

2. 安装ppdet，以及所需依赖包。
```shell script
pip install ./ppdet-2.0rc0-py3-none-any.whl
```

3. 准备数据，默认使用的是VOC格式的数据集，如果要修改为COCO格式的数据集，需要修改`configs/ppyolo.yml`。其中VOC格式操作如下，首先将标注文件放在`dataset/annotation/`，图像文件放在`dataset/images/`，最后执行`create_data_list.py`程序生成数据列表和类别名称。

4. 修改`configs/ppyolo.yml`中的类别数量`num_classes`，这个类别数量不用算上背景这一类别。

5. 执行`train.py`开始训练。

6. 执行`visualdl --logdir=logs`查看训练可视化信息。


## 评估

执行`eval.py`完成模型评估，其中参数`--weights`为模型的路径，不需要带后缀名，`--overlap_thresh`为需要评分的阈值，执行之后输入如下，从输出可以知道当阈值为0.75时，mAP为90.75。
```
$ python eval.py --weights=save_models/model_final --overlap_thresh=0.75

2020-08-17 12:56:58,530-INFO: Test iter 0
2020-08-17 12:57:13,349-INFO: Test iter 100
2020-08-17 12:57:27,421-INFO: Test iter 200
2020-08-17 12:57:36,716-INFO: Test finish iter 268
2020-08-17 12:57:36,716-INFO: Total number of images: 1070, inference time: 27.43800773116421 fps.
2020-08-17 12:57:36,717-INFO: Start evaluate...
2020-08-17 12:57:36,999-INFO: Accumulating evaluatation results...
2020-08-17 12:57:37,017-INFO: mAP(0.75, 11point) = 90.75
```

## 预测

1. 执行预测执行先要导出预测模型，执行`export_model.py`，其中参数`--weights`为模型的路径，不需要带后缀名，`--output_dir`输出预测模型的路径，文件目录结构如下。

```shell script
$ python export_model.py --weights=save_models/model_final --output_dir=output/

$ tree output
output/
├── infer_cfg.yml
├── __model__
└── __params__
```

2. 项目提供了三种预测方式，第一种`infer.py`是使用预测引擎，预测速度更快，更适合部署在服务端。第二个`infer_path.py`为使用图像路径进行预测。第三个`infer_camera.py`为调用摄像头进行实时预测。