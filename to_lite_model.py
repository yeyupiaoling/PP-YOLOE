import paddlelite.lite as lite

opt = lite.Opt()

opt.set_model_file("output_inference/PPYOLOE_S/model.pdmodel")
opt.set_param_file("output_inference/PPYOLOE_S/model.pdiparams")

opt.set_optimize_out("output_inference/detect_model")
opt.set_model_type("naive_buffer")
opt.set_valid_places("arm")
opt.set_quant_model(True)
opt.set_quant_type("QUANT_INT16")

opt.run()
