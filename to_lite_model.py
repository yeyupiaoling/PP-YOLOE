from paddlelite.lite import *

# 1. 创建opt实例
opt = Opt()
# 2. 指定输入模型地址
opt.set_model_file("models/__model__")
opt.set_param_file("models/__params__")
# 3. 指定转化类型： arm、x86、opencl、xpu、npu
opt.set_valid_places("arm")
# 4. 指定模型转化类型： naive_buffer、protobuf
opt.set_model_type("naive_buffer")
# 4. 输出模型地址
opt.set_optimize_out("models/model")
# 5. 执行模型优化
opt.run()
