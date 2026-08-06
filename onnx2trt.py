import tensorrt as trt

onnx_model_path = "./savemodel/model.onnx"
trt_model_path = "./savemodel/model.trt"

# Create a TensorRT logger
logger = trt.Logger(trt.Logger.WARNING)
explicit_batch = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

if __name__ == '__main__':
    # Create a builder, network, and parser
    with trt.Builder(logger) as builder, \
    builder.create_network(explicit_batch) as network, \
        trt.OnnxParser(network, logger) as parser:

        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                # Without this the script happily writes a broken engine.
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError(f"failed to parse {onnx_model_path}")

        # ponytail: no optimization profile, the ONNX graph has fixed shapes.
        # Add one (with set_shape per input) only if torch2onnx starts exporting dynamic_axes.
        config = builder.create_builder_config()

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT failed to build the engine")

        with open(trt_model_path, "wb") as f:
            f.write(engine)

    print(f"ONNX model has been successfully converted to TensorRT model and saved as {trt_model_path}")
