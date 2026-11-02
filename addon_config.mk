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
	# Windows x64: onnxruntime-win-x64-gpu_cuda13-1.24.4 (CPU, or CUDA with Settings::useCuda - see ofxOnnx.h).
	# onnxruntime.dll and the providers shim are copied next to the exe; the CUDA provider (211 MB, not in git) is
	# fetched by scripts/get_cuda_provider.ps1 and must be copied next to the exe too for CUDA.
	ADDON_INCLUDES = src
	ADDON_INCLUDES += libs/onnxruntime/include
	ADDON_INCLUDES_EXCLUDE = libs/onnxruntime/include/cuda/
	ADDON_INCLUDES_EXCLUDE += libs/onnxruntime/include/onnxruntime/core/providers/cuda
	ADDON_LIBS = libs/onnxruntime/lib/vs/x64/onnxruntime.lib
	ADDON_DLLS_TO_COPY = libs/onnxruntime/lib/vs/x64/onnxruntime.dll
	ADDON_DLLS_TO_COPY += libs/onnxruntime/lib/vs/x64/onnxruntime_providers_shared.dll
