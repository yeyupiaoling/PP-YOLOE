import paddlelite.lite as lite

opt = lite.Opt()

opt.set_model_file("output_inference/ppyolo_mbv3_large_qat/model.pdmodel")
opt.set_param_file("output_inference/ppyolo_mbv3_large_qat/model.pdiparams")

opt.set_optimize_out("detect_model")
opt.set_model_type("naive_buffer")
opt.set_valid_places("arm")

opt.run()
