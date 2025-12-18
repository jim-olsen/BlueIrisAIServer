import onnxruntime as ort, sys
print("Python:", sys.version)
print("ORT:", ort.__version__)
print("Available providers:", ort.get_available_providers())

so = ort.SessionOptions()
so.log_severity_level = 0  # VERBOSE
so.log_verbosity_level = 1
session = ort.InferenceSession(
    r"models\IPcam-combined.onnx",
    sess_options=so,
    providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"device_type":"CPU"}, {}]
)
print("session.get_providers():", session.get_providers())
print("session.get_provider_options():", session.get_provider_options())