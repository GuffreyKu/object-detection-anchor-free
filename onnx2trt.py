import tensorrt as trt
import onnx

onnx_model_path = "./savemodel/model.onnx"
trt_model_path = "./savemodel/model.trt"

# Create a TensorRT logger
logger = trt.Logger(trt.Logger.WARNING)
explicit_batch = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
if __name__ == '__main__':

    

    # # Create a builder, network, and parser
    with trt.Builder(logger) as builder, \
    builder.create_network(explicit_batch) as network, \
        trt.OnnxParser(network, logger) as parser:

        with open(onnx_model_path, 'rb') as model:
            parser.parse(model.read())

        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        with open(trt_model_path, "wb") as f:
            f.write(engine)
   

    print(f"ONNX model has been successfully converted to TensorRT model and saved as {trt_model_path}")