import tensorrt as trt

onnx_file = "MobileNet-v2.onnx"
engine_file = "MobileNet-v2.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)

# EXPLICIT_BATCH Network
explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(explicit_batch)

parser = trt.OnnxParser(network, logger)

print("Parsing ONNX...")
with open(onnx_file, "rb") as f:
    if not parser.parse(f.read()):
        print("ERROR: Failed to parse the ONNX file")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit()

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1 GB workspace

print("Building TensorRT engine...")
engine = builder.build_engine(network, config)

if engine is None:
    print("ERROR: Engine creation failed")
    exit()

print("Serializing engine...")
serialized_engine = engine.serialize()

with open(engine_file, "wb") as f:
    f.write(serialized_engine)

print("Done! Saved engine as:", engine_file)
