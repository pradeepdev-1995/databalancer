from fastT5 import OnnxT5, get_onnx_runtime_sessions,generate_onnx_representation, quantize

def quantizeModel(model_or_model_path):

    onnx_model_paths                = generate_onnx_representation(model_or_model_path)
    quant_model_paths               = quantize(onnx_model_paths)
    model_sessions                  = get_onnx_runtime_sessions(quant_model_paths)
    quantizedModel                  = OnnxT5(model_or_model_path, model_sessions)

    return quantizedModel