import datetime
from datetime import UTC

from flask import Flask, request, jsonify, g
import os
import glob
import torch
import onnxruntime as ort
import io
import time
import json
import ast
import uuid
from typing import Tuple, List, Dict, Optional
import logging
import platform

try:
    from PIL import Image
    import numpy as np
except Exception:
    Image = None
    np = None

SERVER_PORT = 5000
if platform.system() == 'Windows':
    try:
        INTEL_OPENVINO_DIR = os.environ.get('INTEL_OPENVINO_DIR')
        os.add_dll_directory(INTEL_OPENVINO_DIR + '\\runtime\\bin\\intel64\\Release')
        os.add_dll_directory(INTEL_OPENVINO_DIR + '\\runtime\\bin\\intel64\\Debug')
        os.add_dll_directory(INTEL_OPENVINO_DIR + '\\runtime\\3rdparty\\tbb\\bin')
    except Exception:
        print("INTEL_OPENVINO_DIR environment variable not set")

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# Store models in dicts keyed by model name (filename without extension)
torch_models = {}
onnx_models = {}
_onnx_names_cache: Dict[str, List[str]] = {}

# COCO80 default class names (used as a fallback when num_classes == 80)
COCO80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
    'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]


def _configure_global_logger() -> logging.Logger:
    """Configure and return a module-level logger.

    Ensures a single StreamHandler with a concise format and INFO level by default.
    """
    logger = logging.getLogger("BlueIrisAIServer")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Avoid double logging via root
        logger.propagate = False
    # Align Flask's app.logger to our logger handlers/level
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)
    app.logger.propagate = False
    return logger


LOGGER = _configure_global_logger()


def _truncate_val(val, max_len: int = 500):
    """
    Truncates a given value to a maximum specified length. Converts the value
    to its string representation if necessary and ensures the length of the
    resulting string does not exceed the limit. Appends a notice of truncation
    if the string is shortened.

    :param val: The value to be truncated. Can be of any type that is convertible
        to a string.
    :param max_len: The maximum allowed length for the string representation
        of the value. Defaults to 500.
    :type max_len: int
    :return: The truncated string representation of the value, or the string
        "<unrepr>" if the value could not be converted to a string.
    :rtype: str
    """
    try:
        s = str(val)
    except Exception:
        return "<unrepr>"
    if len(s) > max_len:
        return s[:max_len] + f"... <truncated {len(s)-max_len} chars>"
    return s


def _collect_request_params() -> Dict[str, object]:
    """Collect non-destructive request params for logging.

    - Does not read file streams; only logs filenames and mimetypes.
    - Includes query args, form fields, and JSON body if parseable.
    """
    # Query and form
    query = {k: _truncate_val(v) for k, v in request.args.items(multi=False)} if request.args else {}
    form = {k: _truncate_val(v) for k, v in request.form.items(multi=False)} if request.form else {}

    # JSON body (silent to avoid raising)
    json_body = None
    try:
        json_obj = request.get_json(silent=True)
        if json_obj is not None:
            # Truncate deeply only at top-level string values
            if isinstance(json_obj, dict):
                json_body = {k: (_truncate_val(v) if isinstance(v, (str, bytes)) else v) for k, v in json_obj.items()}
            else:
                json_body = json_obj
    except Exception:
        json_body = "<unparsable>"

    # Files metadata
    files = {}
    try:
        for k, storage in (request.files or {}).items():
            files[k] = {
                "filename": getattr(storage, "filename", None),
                "content_type": getattr(storage, "mimetype", None),
                # content_length may be None; we won't read streams to compute size
                "content_length": getattr(storage, "content_length", None),
            }
    except Exception:
        files = {"_error": "<files unavailable>"}

    return {
        "query": query,
        "form": form,
        "json": json_body,
        "files": files,
    }


@app.before_request
def log_incoming_request():
    """
    Marks the start time of the incoming request for duration logging and logs the
    request details such as method, URL, path, remote address, and parameters.

    This function is executed before handling each HTTP request in the application
    to gather and log relevant information for debugging and monitoring purposes.

    :raises Exception: Logs a warning if an error occurs during the logging of the
        request details.
    """
    # Mark start time for duration logging
    g._req_start_ts = time.time()
    try:
        info = {
            "method": request.method,
            "url": request.url,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "params": _collect_request_params(),
        }
        LOGGER.info("Incoming request: %s", json.dumps(info, default=str))
    except Exception as e:
        LOGGER.warning("Failed to log request: %s", e)


@app.after_request
def log_outgoing_response(response):
    """
    Logs details about the outgoing HTTP response, including the request method,
    path, response status code, and the time taken to process the request in
    milliseconds if the timing information is available.

    If an exception is raised during the logging process, it is caught and
    silently ignored to ensure the response is returned to the client
    without interruption.

    :param response: The HTTP response object being sent back to the client.
    :type response: flask.wrappers.Response
    :return: The original HTTP response object to be returned to the client.
    :rtype: flask.wrappers.Response
    """
    try:
        start = getattr(g, "_req_start_ts", None)
        dur_ms = (time.time() - start) * 1000.0 if start is not None else None
        LOGGER.info(
            "Request done: %s %s -> %s%s",
            request.method,
            request.path,
            response.status_code,
            f" in {dur_ms:.1f} ms" if dur_ms is not None else "",
        )
    except Exception:
        pass
    return response


def select_onnx_providers() -> Tuple[List[str], Optional[List[Dict[str, str]]]]:
    """Choose best available ONNX Runtime execution providers, preferring GPU.

    Preference order:
      1) CUDAExecutionProvider (NVIDIA)
      2) ROCMExecutionProvider (AMD on Linux)
      3) DmlExecutionProvider (DirectML on Windows)
      4) OpenVINOExecutionProvider (Intel; we try GPU FP16 when possible)
      5) CPUExecutionProvider

    Returns a (providers, provider_options) tuple suitable for InferenceSession().
    provider_options may be None when not needed.
    """
    try:
        avail = ort.get_available_providers() or []
    except Exception:
        avail = []

    preferred = [
        'CUDAExecutionProvider',
        'ROCMExecutionProvider',
        'DmlExecutionProvider',
        'OpenVINOExecutionProvider',
        'CPUExecutionProvider'
    ]

    providers: List[str] = [p for p in preferred if p in avail]
    if not providers:
        providers = ['CPUExecutionProvider']

    provider_options: Optional[List[Dict[str, str]]] = None

    # When OpenVINO is the first choice, prefer GPU FP16 if available.
    # This only applies when using the OpenVINO EP.
    if providers and providers[0] == 'OpenVINOExecutionProvider':
        # Common OpenVINO EP option keys: device_type (e.g., 'GPU_FP16', 'GPU', 'CPU')
        # When passing provider_options alongside providers, onnxruntime requires both lists
        # to be the same length. Provide an empty dict for providers that don't need options.
        first_opts: Dict[str, str] = {'device_type': 'GPU_FP16'}
        provider_options = [first_opts] + [{}] * (len(providers) - 1)

    # When CUDA/ROCM/DML are first, we don't need extra options by default.
    return providers, provider_options


@app.route('/')
def index():
    """
    Serve the static HTML file instead of hard-coded HTML

    :return: The contents of the index.html file
    """
    return app.send_static_file('index.html')

def init():
    """
    Initializes global dictionaries torch_models and onnx_models by loading machine learning
    models from 'models' directory. It scans the directory for YOLOv5 `.pt` files and `.onnx`
    files, processes them with appropriate frameworks (`torch.hub` for `.pt` files and
    `onnxruntime` for `.onnx` files), and stores the loaded models in the global dictionaries
    with normalized model names.

    :global dict torch_models: A dictionary containing loaded YOLOv5 PyTorch models. The keys
        are normalized model names in lowercase, and the values are the corresponding PyTorch
        model instances.
    :global dict onnx_models: A dictionary containing loaded ONNX models. The keys are normalized
        model names in lowercase, and the values are the corresponding ONNX runtime session
        objects.

    """
    global torch_models, onnx_models

    models_dir = 'models'

    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' does not exist")
        return

    # Find and load YOLOv5 .pt files via torch.hub (avoids models.yolo import error)
    pt_files = glob.glob(os.path.join(models_dir, '*.pt'))
    for pt_file in pt_files:
        try:
            # Load YOLOv5 custom weights; torch.hub fetches ultralytics/yolov5 repo if needed
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_file)
            # Always normalize model names to lower-case
            model_name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            torch_models[model_name] = model
            print(f"Loaded YOLOv5 PyTorch model '{model_name}' from: {pt_file}")
        except Exception as e:
            print(f"Error loading YOLOv5 model {pt_file}: {e}")

    # Find and load .onnx files
    onnx_files = glob.glob(os.path.join(models_dir, '*.onnx'))
    for onnx_file in onnx_files:
        try:
            providers, provider_options = select_onnx_providers()
            # onnxruntime expects keyword 'providers' and optional 'provider_options'
            if provider_options is None:
                session = ort.InferenceSession(onnx_file, providers=providers)
            else:
                so = ort.SessionOptions()
                so.log_severity_level = 0
                session = ort.InferenceSession(onnx_file, sess_options=so, providers=providers, provider_options=provider_options)
            # Always normalize model names to lower-case
            model_name = os.path.splitext(os.path.basename(onnx_file))[0].lower()
            # Attach model name to session so we can resolve class names later
            try:
                setattr(session, '_bi_model_name', model_name)
                # Also remember what providers/options we asked for so we can
                # use them for GPU detection when runtime doesn't expose them.
                setattr(session, '_bi_init_providers', providers)
                setattr(session, '_bi_init_provider_options', provider_options)
            except Exception:
                pass
            onnx_models[model_name] = session
            try:
                used_providers = session.get_providers()
            except Exception:
                used_providers = providers
            print(f"Loaded ONNX model '{model_name}' from: {onnx_file} | providers={used_providers}")
        except Exception as e:
            print(f"Error loading ONNX model {onnx_file}: {e}")


@app.route('/v1/vision/custom/list', methods=['GET', 'POST'])
def list_custom_models():
    # Return the list of available model names from both frameworks in the same format
    # as used by codeproject ai
    names = sorted(set([str(n).lower() for n in torch_models.keys()] + [str(n).lower() for n in onnx_models.keys()]))
    return jsonify({"success": True, 
                    "models": names,
                    "inferenceMs" : 0,
                    "processMs" : 0,
                    "analysisRoundTripMs": 1,
                    "moduleName": "Object Detection (YOLOv5)",
                    "moduleId": str(uuid.uuid4()),
                    "command": "list-custom",
                    "requestId": str(uuid.uuid4()),
                    "processedBy": "localhost",
                    "timeStampUTC": datetime.datetime.now(UTC).isoformat()})


@app.route('/models')
def list_models():
    # Return the list of available model names from both frameworks
    # Ensure model names are lower-case in the response
    names = sorted(set([str(n).lower() for n in torch_models.keys()] + [str(n).lower() for n in onnx_models.keys()]))
    return jsonify({"models": names})

def letterbox(im: Image.Image, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    Adjust an image to fit within a specified rectangular shape while maintaining its original
    aspect ratio. The function resizes the image and places it within a new canvas of the specified
    dimensions, padding empty spaces with the given color.

    :param im: The input image to be resized and padded.
    :param new_shape: Target shape (width, height) for the resized and padded image.
    :param color: RGB color tuple to fill the padding areas of the new canvas.

    :return: A tuple containing:
        - The processed image with the new size and proper padding.
        - The resizing scale factor used to fit the original image within the target shape.
        - A tuple with the number of pixels padded on the left and top, respectively.
    """
    w0, h0 = im.size
    r = min(new_shape[0] / w0, new_shape[1] / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    im = im.resize(new_unpad, Image.BILINEAR)
    dw = new_shape[0] - new_unpad[0]
    dh = new_shape[1] - new_unpad[1]
    pad_left = dw // 2
    pad_top = dh // 2
    new_im = Image.new('RGB', new_shape, color)
    new_im.paste(im, (pad_left, pad_top))
    return new_im, r, (pad_left, pad_top)


def yolov5_onnx_infer(session: ort.InferenceSession, image_bytes: bytes, conf_thres: float, iou_thres: float, max_det: int) -> List[Dict]:
    """
    Performs object detection inference using a YOLOv5 ONNX model. This function processes an
    input image, applies the model for detecting objects, filters the results based on confidence
    and performs non-maximum suppression (NMS). The result is a list of detections containing
    object classes and corresponding bounding boxes mapped to the original image dimensions.

    :param session: The ONNX runtime inference session for executing the YOLOv5 model.
                    It encapsulates the pre-loaded model ready for inference.
    :type session: ort.InferenceSession
    :param image_bytes: Byte data of the input image. It is expected to be in a supported image format (e.g., PNG, JPEG).
    :type image_bytes: bytes
    :param conf_thres: The confidence threshold. Detections with confidence below this value are omitted.
    :type conf_thres: float
    :param iou_thres: The Intersection Over Union (IoU) threshold for non-maximum suppression.
                      Detections with IoU above this threshold across classes are suppressed.
    :type iou_thres: float
    :param max_det: The maximum number of detections to retain after NMS.
    :type max_det: int
    :return: A list of detection dictionaries, each containing:
             - class (int): The class ID of the detection.
             - label (str): The label or name of the detected class.
             - confidence (float): The confidence score of the detection.
             - box (dict): The bounding box coordinates (`x1`, `y1`, `x2`, `y2`) of the detection in the original image space.
    :rtype: List[Dict]
    """
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required for ONNX inference")

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_lb, r, (pad_w, pad_h) = letterbox(img, (640, 640))
    im_arr = np.array(img_lb, dtype=np.float32) / 255.0  # HWC, [0,1]
    im_arr = np.transpose(im_arr, (2, 0, 1))  # CHW
    im_arr = np.expand_dims(im_arr, 0)  # NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: im_arr})
    preds = outputs[0]
    # preds shape: (1, N, 85) -> [x,y,w,h,conf, 80 class confs]
    preds = np.squeeze(preds, axis=0)

    # Convert to xyxy
    boxes_xywh = preds[:, 0:4]
    conf_obj = preds[:, 4:5]
    cls_conf = preds[:, 5:]
    cls_ids = np.argmax(cls_conf, axis=1)
    cls_scores = cls_conf[np.arange(cls_conf.shape[0]), cls_ids]
    scores = conf_obj.flatten() * cls_scores

    # Filter by confidence
    mask = scores >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if boxes_xywh.size == 0:
        return []

    # xywh (center-based 640 space) -> xyxy in 640 space
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    keep = nms_numpy(boxes_xyxy, scores, iou_thres)
    keep = keep[:max_det]

    # Resolve class names (cache per ONNX model name when possible)
    try:
        num_classes = cls_conf.shape[1]
    except Exception:
        num_classes = None

    # We will try to pull names from model metadata, sidecar files, or fallbacks
    model_name = getattr(session, '_bi_model_name', None)
    names = resolve_onnx_class_names(session, model_name, num_classes)

    detections = []
    for i in keep:
        bx = boxes_xyxy[i]
        score = float(scores[i])
        cid = int(cls_ids[i])
        # Map back to original resolution
        # First, remove padding, then scale by 1/r
        bx[[0, 2]] -= pad_w
        bx[[1, 3]] -= pad_h
        bx = bx / r
        label = names[cid] if names and 0 <= cid < len(names) else str(cid)
        detections.append({
            "class": cid,
            "label": label,
            "confidence": round(score, 6),
            "box": {"x1": int(max(0, bx[0])), "y1": int(max(0, bx[1])), "x2": int(max(0, bx[2])), "y2": int(max(0, bx[3]))}
        })
    return detections


def nms_numpy(boxes: 'np.ndarray', scores: 'np.ndarray', iou_thres: float) -> List[int]:
    """
    Performs Non-Maximum Suppression (NMS) on the set of bounding boxes and scores. This operation
    filters out overlapping bounding boxes based on their Intersection over Union (IoU) scores,
    keeping the bounding boxes with the highest scores while removing others that have a high
    overlap.

    :param boxes: A 2D array of shape (N, 4) where N is the number of bounding boxes. Each bounding
        box is represented as [x1, y1, x2, y2], defining the top-left and bottom-right coordinates
        of the box.
    :param scores: A 1D array of shape (N,) containing the scores for each bounding box, indicating
        the confidence of the detection.
    :param iou_thres: A float value representing the IoU threshold. Any bounding boxes with an IoU
        greater than this value will be suppressed.
    :return: A list of indices representing the remaining bounding boxes after applying NMS. The
        indices correspond to the positions in the original input `boxes` and `scores` lists.
    """
    # boxes: (N,4) [x1,y1,x2,y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def torch_infer(model, image_bytes: bytes, conf: float, iou: float, max_det: int) -> List[Dict]:
    """
    Runs inference on an input image using a YOLOv5 model and returns
    the detected objects as a list of dictionaries. The function allows
    optional customization of detection parameters such as confidence
    threshold, IoU threshold, and the maximum number of detections.

    :param model: The YOLOv5 model instance used for running inference.
                  It should be compatible with AutoShape and have configurable
                  attributes such as confidence, IoU threshold, and maximum
                  detections.
    :type model: Any
    :param image_bytes: Binary data of the input image. Can be in formats
                        supported by PIL (e.g., JPEG, PNG). If PIL is not
                        available, raw bytes are passed to the model.
    :type image_bytes: bytes
    :param conf: Confidence threshold for detection filtering. Only
                 objects with confidence scores above this value will
                 be returned.
    :type conf: float
    :param iou: Intersection over Union (IoU) threshold for non-maximum
                suppression. Higher values result in stricter bounding
                box filtering.
    :type iou: float
    :param max_det: Maximum number of detections to include in the output.
    :type max_det: int
    :return: A list of dictionaries, each representing a detected object.
             Each dictionary contains the following keys:
             - "class": Integer index of the detected class.
             - "label": String name of the detected class, if available.
             - "confidence": Confidence score of the detection (float).
             - "box": A dictionary with bounding box coordinates:
               * "x1": Top-left x-coordinate (int).
               * "y1": Top-left y-coordinate (int).
               * "x2": Bottom-right x-coordinate (int).
               * "y2": Bottom-right y-coordinate (int).
    :rtype: List[Dict]
    """
    # Use PIL if available to ensure consistency
    if Image is not None:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    else:
        # Fallback: pass raw bytes, YOLOv5 can accept bytes
        img = image_bytes

    # Configure YOLOv5 AutoShape parameters via attributes (avoids unexpected kwargs)
    try:
        if conf is not None:
            # Support both modern and older attribute names
            if hasattr(model, 'conf'):
                setattr(model, 'conf', float(conf))
            elif hasattr(model, 'conf_thres'):
                setattr(model, 'conf_thres', float(conf))
        if iou is not None:
            if hasattr(model, 'iou'):
                setattr(model, 'iou', float(iou))
            elif hasattr(model, 'iou_thres'):
                setattr(model, 'iou_thres', float(iou))
        if max_det is not None and hasattr(model, 'max_det'):
            setattr(model, 'max_det', int(max_det))
    except Exception:
        # Best-effort; continue with defaults if attributes are not present
        pass

    # Call without unsupported kwargs to avoid AutoShape.forward() errors
    results = model(img, size=640)
    names = model.names if hasattr(model, 'names') else None
    det = []
    # results.xyxy[0]: [x1,y1,x2,y2,conf,cls]
    for *xyxy, confv, cls in results.xyxy[0].tolist():
        cid = int(cls)
        # Resolve label from model.names which may be a list/tuple or dict
        label = None
        try:
            if isinstance(names, dict):
                # YOLOv5 often exposes names as dict {int: str} (or keys as strings)
                label = names.get(cid) or names.get(str(cid))
            elif isinstance(names, (list, tuple)):
                if 0 <= cid < len(names):
                    label = names[cid]
        except Exception:
            label = None
        if label is None:
            label = str(cid)
        det.append({
            "class": cid,
            "label": label,
            "confidence": round(float(confv), 6),
            "box": {"x1": int(xyxy[0]), "y1": int(xyxy[1]), "x2": int(xyxy[2]), "y2": int(xyxy[3])}
        })
    return det[:max_det]


def friendly_provider_name(provider: str) -> str:
    """
    Maps ONNX Runtime provider identifiers to a user-friendly name. This function
    takes a provider identifier and maps it to a readable and expected name that
    clients can easily recognize. For example, identifiers containing "cuda" are
    mapped to "GPU", and those with "dml" or "directml" are mapped to "DirectML".
    If no match is found, the function defaults to returning "CPU".

    :param provider: The ONNX Runtime provider identifier as a string.
    :return: A user-friendly string name associated with the provider.
    """
    # Map ONNX Runtime provider identifiers to friendly names expected by clients
    p = (provider or '').lower()
    if 'cuda' in p:
        return 'GPU'
    if 'dml' in p or 'directml' in p:
        return 'DirectML'
    if 'openvino' in p:
        return 'OPENVINO'
    if 'tensorrt' in p:
        return 'TensorRT'
    if 'coreml' in p:
        return 'CoreML'
    return 'CPU'

def onnx_exec_provider_and_gpu(session: 'ort.InferenceSession') -> Tuple[str, bool]:
    """
    Determines the execution provider and whether GPU is available for the given ONNX Runtime
    InferenceSession.

    This function evaluates the available execution providers in the given session and attempts
    to choose the most optimal one based on a predefined preference order. It also determines
    if the chosen execution provider supports GPU execution.

    :param session: An instance of ``ort.InferenceSession`` from ONNX Runtime.
    :return: A tuple containing:
        - The friendly name of the chosen execution provider (str).
        - A boolean indicating whether the execution provider supports GPU (bool).
    """
    try:
        providers = session.get_providers() or []
    except Exception:
        providers = []

    # Pick the strongest EP that is present, instead of blindly taking providers[0]
    preference = [
        'CUDAExecutionProvider',
        'ROCMExecutionProvider',
        'DmlExecutionProvider',
        'OpenVINOExecutionProvider',
        'CPUExecutionProvider'
    ]

    chosen_raw = None
    for p in preference:
        if p in providers:
            chosen_raw = p
            break
    if not chosen_raw:
        chosen_raw = providers[0] if providers else 'CPUExecutionProvider'

    # Determine can_gpu
    can_gpu = False
    if chosen_raw in ('CUDAExecutionProvider', 'ROCMExecutionProvider', 'DmlExecutionProvider'):
        can_gpu = True
    elif chosen_raw == 'OpenVINOExecutionProvider':
        # Try to read OpenVINO device_type from provider options.
        # ORT may return provider options in various shapes across versions.
        def extract_openvino_opts_from(opts_obj, prov_list):
            # Returns a dict of OV options or None.
            if not opts_obj:
                return None
            # Case 1: mapping provider->options
            if isinstance(opts_obj, dict):
                ov = opts_obj.get('OpenVINOExecutionProvider')
                if isinstance(ov, dict):
                    return ov
                # Some builds may store a flat dict if only one provider used
                if 'device_type' in opts_obj or 'device_id' in opts_obj:
                    return opts_obj
                return None
            # Case 2: sequence aligned with providers
            if isinstance(opts_obj, (list, tuple)) and prov_list:
                try:
                    idx = prov_list.index('OpenVINOExecutionProvider')
                except Exception:
                    idx = 0
                if 0 <= idx < len(opts_obj):
                    ov = opts_obj[idx]
                    if isinstance(ov, dict):
                        return ov
                # Fallback: if single dict in list
                if len(opts_obj) == 1 and isinstance(opts_obj[0], dict):
                    return opts_obj[0]
            return None

        device_type = ''
        try:
            prov_list = session.get_providers() or []
        except Exception:
            prov_list = []

        # 1) Runtime-reported options
        ov_opts = None
        try:
            runtime_opts = session.get_provider_options()
            ov_opts = extract_openvino_opts_from(runtime_opts, prov_list)
        except Exception:
            ov_opts = None

        # 2) Fallback to options used at init time (attached in init())
        if not ov_opts:
            try:
                init_opts = getattr(session, '_bi_init_provider_options', None)
                init_prov = getattr(session, '_bi_init_providers', prov_list)
                ov_opts = extract_openvino_opts_from(init_opts, init_prov)
            except Exception:
                ov_opts = None

        # 3) As last resort, check env var commonly used to pick device
        if not ov_opts:
            env_dev = os.environ.get('OV_DEVICE') or os.environ.get('OPENVINO_DEVICE') or ''
            device_type = str(env_dev)
        else:
            device_type = str(ov_opts.get('device_type') or ov_opts.get('device_id') or '')

        can_gpu = 'gpu' in device_type.lower()

    return friendly_provider_name(chosen_raw), bool(can_gpu)


def format_predictions_from_detections(detections: List[Dict]) -> List[Dict]:
    """
    Formats raw detection data into a structured list of predictions. Each detection is converted
    to a dictionary containing specific keys such as bounding box coordinates, label, and confidence.
    Default values are used if any key is missing in the detection data.

    :param detections: A list of dictionaries, where each dictionary represents a detection result.
        Each detection is expected to contain the keys `box` (a dictionary containing bounding box
        coordinates `x1`, `y1`, `x2`, `y2`), `label` (a string label or class name), `class` (an
        alternative key for the label), and `confidence` (a numeric confidence score).
    :return: A list of dictionaries representing formatted predictions. Each dictionary contains
        the keys `x_min`, `y_min`, `x_max`, `y_max`, `label`, and `confidence`, representing
        bounding box coordinates, the associated label, and the confidence score, respectively.
    """
    preds = []
    for d in detections:
        box = d.get('box', {})
        preds.append({
            "x_min": int(box.get('x1', 0)),
            "y_min": int(box.get('y1', 0)),
            "x_max": int(box.get('x2', 0)),
            "y_max": int(box.get('y2', 0)),
            "label": d.get('label', str(d.get('class', '0'))),
            "confidence": float(d.get('confidence', 0.0))
        })
    return preds


def parse_names_string(val: str) -> Optional[List[str]]:
    """
    Parses a string containing names into a list of strings. The input string may represent
    a JSON object, a JSON list, or a Python literal dictionary or list-like format. When a dictionary
    is parsed, its values are extracted and sorted by their corresponding integer keys. If the input
    string cannot be parsed into any of the accepted formats, the function returns None.

    :param val: The string representation to parse. Can be JSON or Python literal format.
    :type val: str
    :return: A list of names as strings if parsing is successful, or None if the input cannot
             be parsed into a valid format.
    :rtype: Optional[List[str]]
    """
    if not val:
        return None
    # Try JSON first
    try:
        j = json.loads(val)
        if isinstance(j, dict):
            # keys may be indices as strings
            items = sorted(((int(k), v) for k, v in j.items()), key=lambda x: x[0])
            return [str(v) for _, v in items]
        if isinstance(j, list):
            return [str(x) for x in j]
    except Exception:
        pass
    # Try Python literal dict format used by some exporters
    try:
        obj = ast.literal_eval(val)
        if isinstance(obj, dict):
            items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
            return [str(v) for _, v in items]
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except Exception:
        pass
    return None


def load_sidecar_names(model_name: Optional[str]) -> Optional[List[str]]:
    """
    Loads a list of sidecar names associated with the specified model name. The method looks
    for a file in the `models` directory matching the model name with one of the extensions:
    `.names`, `.labels`, or `.txt`. If a file is found, it reads and parses the non-empty lines
    into a list of strings. If no matching files are found, or the model name is not provided,
    the method returns None.

    :param model_name: The name of the model whose sidecar names are to be loaded. If None, the
                       function will return None.
    :type model_name: Optional[str]
    :return: A list of strings containing non-empty lines from the first matching file, or None
             if no file is found or `model_name` is None.
    :rtype: Optional[List[str]]
    """
    if not model_name:
        return None
    base = os.path.join('models', model_name)
    candidates = [base + ext for ext in ('.names', '.labels', '.txt')]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    lines = [ln.strip() for ln in f.readlines()]
                    lines = [ln for ln in lines if ln]
                    if lines:
                        return lines
            except Exception:
                continue
    return None


def resolve_onnx_class_names(session: ort.InferenceSession, model_name: Optional[str], num_classes: Optional[int]) -> Optional[List[str]]:
    """
    Resolves class names for an ONNX model, leveraging various sources including model
    metadata, sidecar files, and defaults. This function attempts to extract human-readable
    class names associated with the model, or generates fallback classes if unavailable.

    :param session: ONNX runtime inference session for the loaded model.
    :param model_name: Optional string representing the name of the ONNX model. This is
        used for caching resolved class names to enhance performance.
    :param num_classes: Optional integer representing the number of classes. If names cannot
        be resolved, this is used to generate fallback numeric class names or match standard
        datasets such as COCO80.
    :return: A list of class names as strings, or None if class names cannot be resolved.
    """
    # Cache by model name if available
    if model_name and model_name in _onnx_names_cache:
        return _onnx_names_cache[model_name]

    names: Optional[List[str]] = None
    # 1) Try ONNX metadata
    try:
        meta = session.get_modelmeta()
        cmap = getattr(meta, 'custom_metadata_map', {}) or {}
        # Common keys: 'names', 'classes', 'labels'
        for k in ('names', 'classes', 'labels'):
            if k in cmap:
                names = parse_names_string(cmap[k])
                if names:
                    break
        # Some exporters might stash in graph_name or description
        if not names:
            for attr in ('graph_name', 'description'):  # best-effort
                val = getattr(meta, attr, None)
                if isinstance(val, str):
                    parsed = parse_names_string(val)
                    if parsed:
                        names = parsed
                        break
    except Exception:
        pass

    # 2) Sidecar file next to model
    if not names:
        names = load_sidecar_names(model_name)

    # 3) COCO80 fallback when counts match
    if not names and num_classes == 80:
        names = COCO80[:]

    # 4) If still unknown but we know the count, generate numeric strings
    if not names and isinstance(num_classes, int) and num_classes > 0:
        names = [str(i) for i in range(num_classes)]

    # Cache
    if model_name and names:
        _onnx_names_cache[model_name] = names
    return names


def error_response(message: str, error: str, module_id: str, module_name: str, command: str,
                   execution_provider: str, can_use_gpu: bool,
                   inference_ms: int = 0, process_ms: int = 0, analysis_round_trip_ms: int = 0):
    """
    Generates a standardized error response in JSON format.

    The function constructs an error response with various details such as error
    message, related module information, timing metrics, GPU capability, and the
    status of the execution. It is designed for easy integration with APIs
    that require structured error outputs.

    :param message: A string containing the error message to be returned in the response.
    :param error: A string describing the specific error encountered by the system.
    :param module_id: A string identifying the unique ID of the module where the error occurred.
    :param module_name: A string representing the name of the module associated with the error.
    :param command: A string specifying the command or operation being executed at the time of the error.
    :param execution_provider: A string that specifies the execution provider being used (e.g., CPU, GPU).
    :param can_use_gpu: A boolean indicating if the GPU was available and could be used during the process.
    :param inference_ms: An optional integer representing the time taken for inference in milliseconds. Default is 0.
    :param process_ms: An optional integer representing the time taken for processing in milliseconds. Default is 0.
    :param analysis_round_trip_ms: An optional integer representing round-trip time for analysis in milliseconds.
        Default is 0.
    :return: A structured JSON response containing error details, input metadata, and timing metrics.
    """
    return jsonify({
        "success": False,
        "message": message,
        "error": error,
        "predictions": [],
        "count": 0,
        "inferenceMs": int(round(inference_ms)),
        "processMs": int(round(process_ms)),
        "moduleId": module_id,
        "moduleName": module_name,
        "command": command,
        "executionProvider": execution_provider,
        "canUseGPU": bool(can_use_gpu),
        "analysisRoundTripMs": int(round(analysis_round_trip_ms))
    })


@app.route('/v1/vision/custom/<model_name>', methods=['POST'])
def vision_custom(model_name):
    """
    Handles custom vision model inference requests. It supports both PyTorch and ONNX frameworks,
    allowing for image detection using pre-loaded models. The endpoint processes the input image,
    runs inference on the selected model, and returns formatted predictions.

    :param str model_name: Name of the model to use for inference. The model should be preloaded into
                           `torch_models` or `onnx_models` dictionaries for PyTorch or ONNX, respectively.

    :raises HTTPException:
        404 Not Found: If the model specified by `model_name` is not available.
        400 Bad Request: If the file part of the request is missing, no file is selected, or invalid
                         parameter types (e.g., `conf`, `iou`, `max_det`) are provided.
        415 Unsupported Media Type: If the provided file is not an allowed MIME type such as `image/jpeg`
                                     or `image/png`.
        500 Internal Server Error: If there's an error during the inference process.

    :return: JSON response containing:
        - `success` (bool): Indicates if the detection process succeeded.
        - `message` (str): A descriptive message about the operation outcome.
        - `predictions` (list[dict]): List of predicted detections with associated details.
        - `count` (int): Number of detections made.
        - `inferenceMs` (int): Time taken for the inference in milliseconds.
        - `processMs` (int): Total time taken to process the request in milliseconds.
        - `moduleId` (str): Identifier of the model used for inference.
        - `moduleName` (str): Name of the module handling the detection.
        - `command` (str): Command being executed. For example, "detect".
        - `executionProvider` (str): Specifies whether CPU or GPU was used for processing.
        - `canUseGPU` (bool): Indicates if GPU can be used for inference.
        - `analysisRoundTripMs` (int): Total round-trip analysis time in milliseconds.

    :rtype: flask.Response
    """
    # Normalize incoming model name to lower-case for consistent addressing
    model_name = (model_name or '').lower()

    # Model selection: prefer PyTorch if exists, else ONNX
    framework = None
    model = None
    if model_name in torch_models:
        model = torch_models[model_name]
        framework = 'torch'
    elif model_name in onnx_models:
        model = onnx_models[model_name]
        framework = 'onnx'
    else:
        # Return in the new structured format
        return error_response(
            message=f"Model '{model_name}' not found",
            error="NotFound",
            module_id=model_name,
            module_name="vision.custom",
            command="detect",
            execution_provider="CPU",
            can_use_gpu=False
        ), 404

    if 'image' not in request.files:
        return error_response(
            message="No file part in the request",
            error="BadRequest",
            module_id=model_name,
            module_name=f"vision.custom.{framework or 'unknown'}",
            command="detect",
            execution_provider="CPU",
            can_use_gpu=False
        ), 400
    file = request.files['image']
    if file.filename == '':
        return error_response(
            message="No selected file",
            error="BadRequest",
            module_id=model_name,
            module_name=f"vision.custom.{framework or 'unknown'}",
            command="detect",
            execution_provider="CPU",
            can_use_gpu=False
        ), 400
    content_type = file.mimetype or ''
    if content_type not in ("image/jpeg", "image/png", ''):
        return error_response(
            message="Unsupported file type. Use image/jpeg or image/png.",
            error="UnsupportedMediaType",
            module_id=model_name,
            module_name=f"vision.custom.{framework or 'unknown'}",
            command="detect",
            execution_provider="CPU",
            can_use_gpu=False
        ), 415

    image_bytes = file.read()

    # Params
    try:
        conf = float(request.values.get('min_confidence', 0.25))
        iou = float(request.values.get('iou', 0.45))
        max_det = int(request.values.get('max_det', 100))
    except Exception:
        return error_response(
            message="Invalid parameter types for conf/iou/max_det",
            error="BadRequest",
            module_id=model_name,
            module_name=f"vision.custom.{framework or 'unknown'}",
            command="detect",
            execution_provider="CPU",
            can_use_gpu=False
        ), 400

    process_start = time.time()
    infer_start = None
    try:
        if framework == 'torch':
            # Determine provider info for torch
            if torch.cuda.is_available():
                exec_provider = 'GPU'
                can_gpu = True
            else:
                exec_provider = 'CPU'
                can_gpu = False

            infer_start = time.time()
            detections = torch_infer(model, image_bytes, conf, iou, max_det)
            inference_ms = (time.time() - infer_start) * 1000.0

        else:
            # ONNX Runtime provider info
            exec_provider, can_gpu = onnx_exec_provider_and_gpu(model)

            infer_start = time.time()
            detections = yolov5_onnx_infer(model, image_bytes, conf, iou, max_det)
            inference_ms = (time.time() - infer_start) * 1000.0
    except Exception as e:
        process_ms = (time.time() - process_start) * 1000.0
        return error_response(
            message="Inference error",
            error=str(e),
            module_id=model_name,
            module_name=f"vision.custom.{framework}",
            command="detect",
            execution_provider=exec_provider if 'exec_provider' in locals() else 'CPU',
            can_use_gpu=can_gpu if 'can_gpu' in locals() else False,
            inference_ms=int(round((time.time() - (infer_start or process_start)) * 1000.0)),
            process_ms=int(round(process_ms)),
            analysis_round_trip_ms=int(round(process_ms))
        ), 500

    process_ms = (time.time() - process_start) * 1000.0

    predictions = format_predictions_from_detections(detections)
    response = {
        "success": True,
        "message": f"Detection succeeded using {framework}",
        "predictions": predictions,
        "count": len(predictions),
        "inferenceMs": int(round(inference_ms)),
        "processMs": int(round(process_ms)),
        "moduleId": model_name,
        "moduleName": f"vision.custom.{framework}",
        "command": "detect",
        "executionProvider": exec_provider,
        "canUseGPU": bool(can_gpu),
        "analysisRoundTripMs": int(round(process_ms))
    }
    return jsonify(response)


def main():
    init()
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=True)


if __name__ == '__main__':
    main()