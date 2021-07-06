# PP-YOLO

PP-YOLO是PaddleDetection优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型，PP-YOLO在COCO test-dev2017数据集上精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。其中还包含了PP-YOLO tiny模型，此模型后量化压缩模型，将模型体积压缩到1.3M，对精度和预测速度基本无影响，在手机上部署，推理速度也是极快的。


## 训练

1. 安装PaddlePaddle和PaddleDetection依赖库。
```shell script
pip install paddlepaddle-gpu==2.1.0.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
pip install paddledet==2.1.0 -i https://mirror.baidu.com/pypi/simple
```

2. 准备数据，默认使用的是VOC格式的数据集，首先将标注文件放在`dataset/annotation/`，图像文件放在`dataset/images/`，最后执行`create_data_list.py`程序生成数据列表和类别名称。

3. 修改`voc.yml`中的类别数量`num_classes`，这个类别数量不用算上背景这一类别。

4. 执行`train.py`开始训练，其中选择PP-YOLO和PP-YOLO tiny模型，并支持量化训练，具体看配置参数。

5. 执行`visualdl --logdir=log`查看训练可视化信息。


## 评估

执行`eval.py`完成模型评估，其中参数`-o weights`为模型的路径，不需要带后缀名，执行之后输入如下。
```
python eval.py -o weights=output/ppyolo_mbv3_large_qat/best_model

2020-08-17 12:56:58,530-INFO: Test iter 0
2020-08-17 12:57:13,349-INFO: Test iter 100
2020-08-17 12:57:27,421-INFO: Test iter 200
2020-08-17 12:57:36,716-INFO: Test finish iter 268
2020-08-17 12:57:36,716-INFO: Total number of images: 1070, inference time: 27.43800773116421 fps.
2020-08-17 12:57:36,717-INFO: Start evaluate...
2020-08-17 12:57:36,999-INFO: Accumulating evaluatation results...
2020-08-17 12:57:37,017-INFO: mAP(0.5, 11point) = 90.75
```

## 预测

1. 执行预测执行先要导出预测模型，执行`export_model.py`，其中参数`--weights`为模型的路径，不需要带后缀名，`--output_dir`输出预测模型的路径，文件目录结构如下。

```shell script
python export_model.py -o weights=output/ppyolo_mbv3_large_qat/best_model
```

2. 导出PaddleLite模型。
```shell
python to_lite_model.py
```

## 预测

项目提供了两种预测方式，第一种`infer.py`为使用图像路径进行预测。第二个`infer_camera.py`为调用摄像头进行实时预测。