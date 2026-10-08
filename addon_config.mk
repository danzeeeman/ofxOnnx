meta:
    ADDON_NAME = ofxOnnx
    ADDON_DESCRIPTION = Interface Microsoft's ONNXRuntime
    ADDON_AUTHOR = Nick Hardeman
    ADDON_TAGS = "onnx" "machine learning" "inference"
    ADDON_URL = 

common:
    # defines that will be passed to the compiler when including this addon
	ADDON_DEFINES = OF_ADDON_HAS_OFX_ONNX
	
osx:
	ADDON_INCLUDES_EXCLUDE = libs/onnxruntime/include/cuda/
	ADDON_INCLUDES_EXCLUDE += libs/onnxruntime/include/onnxruntime/core/providers/cuda
	ADDON_LDFLAGS=-Wl,-rpath,@executable_path/

linux64:
    ADDON_LDFLAGS += -L/usr/local/cuda/lib64
    ADDON_LDFLAGS += -lcudart
    ADDON_INCLUDES_EXCLUDE = libs/onnxruntime/coreml_provider_factory.h
    ADDON_INCLUDES = src
    ADDON_INCLUDES += libs/onnxruntime/include
    ADDON_CFLAGS += -I/usr/local/cuda/include
    ADDON_DEFINES += OFX_ONNX_USE_CUDA

vs:
	# Windows x64, CPU. onnxruntime.dll (and the providers shim it loads) are copied next to the exe.
	ADDON_INCLUDES = src
	ADDON_INCLUDES += libs/onnxruntime/include
	ADDON_INCLUDES_EXCLUDE = libs/onnxruntime/include/cuda/
	ADDON_INCLUDES_EXCLUDE += libs/onnxruntime/include/onnxruntime/core/providers/cuda
	ADDON_LIBS = libs/onnxruntime/lib/vs/x64/onnxruntime.lib
	ADDON_DLLS_TO_COPY = libs/onnxruntime/lib/vs/x64/onnxruntime.dll
	ADDON_DLLS_TO_COPY += libs/onnxruntime/lib/vs/x64/onnxruntime_providers_shared.dll
