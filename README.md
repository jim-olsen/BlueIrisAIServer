# Blue Iris AI Server
This is a simple python based server that is compatible with the basic vision
detection of the Blue Iris AI image detection support.  It is designed to
serve 'custom' models for image detection of either pytorch (.pt) model
files or onnx (.onnx) model files.  The model files are simply placed into
the 'models' directory and will be served as a custom model that is the same
name as the file.  This is a replacement for the codeproject.ai server for
basic image detection.

I had a lot of problems with the directml support crashing codeproject.ai
and developed this to be a simple replacement.  Pytorch files will support
a cuda environment.  To take advantage of the gpu on intel based hardware,
you can use onnx files with openvino.  To do this you must install the
openvino runtime.  Instructions included below.

Simply start the python server by running server.py.  Point your blueiris
configuration to the address of the machine (can be run on the same machine)
with port 5000.  Blueiris should then work as normal.

## Installing OpenVINO on Windows
