import paddle2onnx

paddle2onnx.export(model_file="output_inference/PPYOLOE_S/model.pdmodel",
                   params_file="output_inference/PPYOLOE_S/model.pdiparams",
                   save_file="output_inference/model.onnx",
                   opset_version=11,
                   auto_upgrade_opset=True,
                   verbose=True,
                   enable_onnx_checker=True,
                   enable_experimental_op=True,
                   enable_optimize=True)
