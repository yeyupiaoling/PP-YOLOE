import paddle2onnx

paddle2onnx.command.program2onnx(model_dir="models/PPYOLOE_M/infer/",
                                 model_filename="model.pdmodel",
                                 params_filename="model.pdiparams",
                                 save_file="models/PPYOLOE_M/model.onnx",
                                 opset_version=11)
