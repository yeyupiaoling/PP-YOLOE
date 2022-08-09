import paddle2onnx

paddle2onnx.command.program2onnx(model_dir="output_inference/PPYOLOE_M/",
                                 model_filename="model.pdmodel",
                                 params_filename="model.pdiparams",
                                 save_file="output_inference/model.onnx",
                                 opset_version=11)
