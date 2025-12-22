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

## Model Notes

Note that I have included to example models in the models directory.  These are just
for demonstration purposes only.  The real supported files are at:

https://github.com/MikeLud/CodeProject.AI-Custom-IPcam-Models

and should be sourced from there directly.  These are included here for convenience.

## Running automated on windows

Note that the easiest way to do this is to create a simple batch file.  Have the batch file
change directory to the project directory under src/main/python and then run the server.py file.
Myself, I setup a virtual environment and use that python instance directly.  I then use task scheduler
and create a new task on startup that runs the batch file.

## Installing OpenVINO on Windows

These are the steps I used to install OpenVINO on Windows 11.

```
mkdir C:\Program Files (x86)\Intel
```

Then download the runtime archive file

```
cd <user_home>\Downloads
curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4/windows/openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64.zip --output openvino_2025.4.0.zip
```

Extract the archive and copy to the target directory

```
tar -xf openvino_2025.4.0.zip
ren openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64 openvino_2025.4.0
move openvino_2025.4.0 "C:\Program Files (x86)\Intel"
cd C:\Program Files (x86)\Intel
mklink /D openvino_2025 openvino_2025.4.0
```

Next you want to setup the environment variables.  You can do this for each session if you like by using
the setupvars.bat file in the bin directory.  I prefer to set them globally so I can use them from any
program or prompt.  These are the required environment variables.

```
INTEL_OPENVINO_DIR=C:\Program Files (x86)\Intel\openvino_2025
PYTHONPATH=%INTEL_OPENVINO_DIR%\python;%INTEL_OPENVINO_DIR%\python\python3
```

Then add on to the path variable the following directories:

```
%INTEL_OPENVINO_DIR%\runtime\bin\intel64\Release
%INTEL_OPENVINO_DIR%\runtime\bin\intel64\Debug
%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\bin
```

The requirements.txt file in the project already loads the python libraries required to utlize openvino
with onnx models.