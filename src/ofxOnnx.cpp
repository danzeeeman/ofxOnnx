#include "ofxOnnx.h"

#if defined(OFX_ONNX_USE_CUDA)
#include <cuda_runtime.h>
#endif

static const char * DTypeName(ONNXTensorElementDataType t) {
	switch (t) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return "float32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return "uint8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return "int8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		return "uint16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return "int16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return "int32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return "int64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		return "string";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		return "bool";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		return "float16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return "float64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		return "uint32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		return "uint64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		return "complex64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
		return "complex128";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		return "bfloat16";
	default:
		return "unknown";
	}
}

//-----------------------------------------------------------------------
bool ofxOnnx::load(ofxOnnx::Settings asettings) {

	auto modelPath = ofToDataPath(asettings.modelPath.string(), true);

	if (!ofFile::doesFileExist(modelPath)) {
		ofLogError("ofxOnnx::load") << "file does not exist: " << modelPath;
		return false;
	}

	mSettings = asettings;
	
	if( mSettings.logLevel == OF_LOG_VERBOSE ) {
		ofLogNotice("ofxOnnx::load") << "Trying to load model from " << modelPath;
	}

	OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_INFO;
	if( mSettings.logLevel == OF_LOG_VERBOSE ) {
		logLevel = ORT_LOGGING_LEVEL_VERBOSE;
	} else if( mSettings.logLevel == OF_LOG_WARNING ) {
		logLevel = ORT_LOGGING_LEVEL_WARNING;
	} else if (mSettings.logLevel == OF_LOG_ERROR) {
		logLevel = ORT_LOGGING_LEVEL_ERROR;
	} else if (mSettings.logLevel == OF_LOG_FATAL_ERROR) {
		logLevel = ORT_LOGGING_LEVEL_FATAL;
	}

	// ORT_LOGGING_LEVEL_WARNING
	mEnv = Ort::Env(logLevel, mSettings.envName.c_str() );
	
	if( mEnv == nullptr ) {
		ofLogError("ofxOnnx::load") << "unable to create env: " << mSettings.envName;
		return false;
	}
	
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(mSettings.numIntraOpThreads);
//	session_options.SetInterOpNumThreads(1);
	session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	
	// Enable flush-to-zero and denormal-as-zero for CPU ops
	// "0" = do NOT disable flushing = flushing IS enabled
	session_options.AddConfigEntry("session.disable_denormal_flushing", "0");
	session_options.AddConfigEntry("session.use_fp16_initializers", "1");

//	std::string model_dir = ofToDataPath(asettings.modelPath.parent_path().string(), true);
//	std::string model_dir = ofToDataPath(asettings.modelPath.string(), true);
	
	// Set the external data directory
//	session_options.AddConfigEntry("session.external_data_path", model_dir.c_str());
//	// Add external initializers path
//	session_options.AddExternalInitializers(
//											{}, // Empty vector for names (loads all external data)
//											model_dir.c_str()
//											);
	

#if defined(OFX_ONNX_USE_CUDA)
	if( mSettings.useCuda ) {
		cudaDeviceProp p {};
		cudaGetDeviceProperties(&p, 0);
		std::cout << "name=" << p.name << " cc=" << p.major << p.minor << "\n";
		std::cout << "ORT VERSION: " << Ort::GetVersionString()<< std::endl; 


		// --- CUDA provider on ---
		OrtCUDAProviderOptions cuda {};
		cuda.device_id = mSettings.deviceId;
		cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda.do_copy_in_default_stream = 1;  // reduces stalls from the 10 memcpy nodes
		cuda.arena_extend_strategy = 0;       // avoids the incremental BFCArena growth you saw
		//cuda.gpu_mem_limit = 512ULL * 1024 * 1024; // pre-size arena

		session_options.AppendExecutionProvider_CUDA(cuda); // Uses device 0 by default
	}
#else
	if(mSettings.useCuda) {
		mSettings.useCuda = false;
		ofLogWarning("ofxOnnx::load") << "Requesting CUDA invalid, must define OFX_ONNX_USE_CUDA and have CUDA installed on Linux or Windows and linked accordingly. See addon_config.mk ";
	}
#endif
	
#if defined(TARGET_OSX)
//	https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
	if( mSettings.useCoreML ) {
		std::unordered_map<std::string, std::string> provider_options;
		provider_options["ModelFormat"] = "MLProgram";
		provider_options["MLComputeUnits"] = "ALL";
		provider_options["RequireStaticInputShapes"] = "0";
		provider_options["EnableOnSubgraphs"] = "0";
//		0: Use float32 data type to accumulate data.
//		1: Use low precision data(float16) to accumulate data.
		provider_options["AllowLowPrecisionAccumulationOnGPU"] = "1";
		session_options.AppendExecutionProvider("CoreML", provider_options);
	}
#else
	mSettings.useCoreML = false;
#endif
	
	// load the model directly. ONNX Runtime wants a native path string: wchar_t on Windows, char elsewhere,
	// which is what std::filesystem::path::c_str() returns on each.
	const of::filesystem::path nativeModelPath(modelPath);
	try {
		mSession = Ort::Session(mEnv, nativeModelPath.c_str(), session_options);
	} catch (const Ort::Exception& e) {
		std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
		std::cerr << "Error Code: " << e.GetOrtErrorCode() << std::endl;
		return false;
	} catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
		return false;
	} catch (...) {
		ofLogError("ofxOnnx::load") << "Error loading from " << modelPath;
		return false;
	}
	
	
//	mSession = Ort::Session(mEnv, modelPath.c_str(), session_options);
	
	if( mSession == nullptr ) {
		ofLogError("ofxOnnx::load") << "unable to create session: " << mSettings.envName;
		return false;
	}
	
//	mSession = std::make_unique<Ort::Session>(mEnv, model_path, session_options);

	mCpuMem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	
	mInputNames.resize( getInputCount() );
	for( size_t i = 0; i < getInputCount(); i++ ) {
		mInputNames[i] = getInputName(i);
	}
	
	mOutputNames.resize(getOutputCount());
	for( size_t i = 0; i < getOutputCount(); i++ ) {
		mOutputNames[i] = getOutputName(i);
	}
	// C-string tables for runFloat(). Taken after the vectors are final: the strings must not move.
	mInputNamePtrs.clear();
	for( auto& s : mInputNames ) mInputNamePtrs.push_back(s.c_str());
	mOutputNamePtrs.clear();
	for( auto& s : mOutputNames ) mOutputNamePtrs.push_back(s.c_str());
	
	if( mSettings.bPrintModelInfo ) {
		std::cout << "--- " << mSettings.envName << " ---" << std::endl;
		printMetaData();
		printInputs();
		printOutputs();
	}

	// Ort::AllocatorWithDefaultOptions alloc;
	// {
	// 	auto n = mSession.GetInputNameAllocated(0, alloc);
	// 	in_name_ = n.get();
	// }
	// {
	// 	auto n = mSession.GetOutputNameAllocated(0, alloc);
	// 	out0_name_ = n.get();
	// } // "dets"
	// {
	// 	auto n = mSession.GetOutputNameAllocated(1, alloc);
	// 	out1_name_ = n.get();
	// } // "keypoints"

	// Cache input shape we’re going to use (NCHW 1x3x640x640 here)
	// auto name = mSession->GetInputNameAllocated(0, alloc);
	// size_t n_in = mSession->GetInputCount();
	// if (mSession->GetInputCount() > 0) {
	// 	Ort::TypeInfo ti = mSession->GetInputTypeInfo(0);
	// 	auto tt = ti.GetTensorTypeAndShapeInfo();
	// 	auto shape = tt.GetShape();
	// 	if (shape.size() >= 4) {
	// 		mInputShape = { 1, shape[1], shape[2], shape[3] };
	// 	}
	// } else {
	// 	mInputShape = { 1, 3, 640, 640 };
	// }

	// mInputBuffer.resize(size_t(mInputShape[0] * mInputShape[1] * mInputShape[2] * mInputShape[3]));

	return true;
}

//--------------------------------------------------------------------
bool ofxOnnx::runFloat(const std::vector<float>& input, std::vector<float>& output) {
	if( !hasSession() ) {
		ofLogError("ofxOnnx::runFloat") << "no session: load() a model first";
		return false;
	}
	if( mInputNamePtrs.size() != 1 || mOutputNamePtrs.size() != 1 ) {
		ofLogError("ofxOnnx::runFloat") << "needs a model with one input and one output, this one has "
										<< mInputNamePtrs.size() << " and " << mOutputNamePtrs.size();
		return false;
	}
	if( getInputElementType(0) != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ) {
		ofLogError("ofxOnnx::runFloat") << "the input is not float32";
		return false;
	}

	// The model's input shape, with dynamic dimensions (-1) taken as 1.
	std::vector<int64_t> shape = getInputShape(0);
	size_t expected = 1;
	for( auto& d : shape ) {
		if( d < 0 ) d = 1;
		expected *= (size_t)d;
	}
	if( input.size() != expected ) {
		ofLogError("ofxOnnx::runFloat") << "the model takes " << expected << " values, got " << input.size();
		return false;
	}

	try {
		// The tensor reads straight from `input` (it is only read; the API takes a non-const pointer).
		Ort::Value in = Ort::Value::CreateTensor<float>(mCpuMem, const_cast<float*>(input.data()), input.size(),
														shape.data(), shape.size());
		std::vector<Ort::Value> out = mSession.Run(Ort::RunOptions{ nullptr },
												   mInputNamePtrs.data(), &in, 1,
												   mOutputNamePtrs.data(), 1);
		if( out.empty() || !out[0].IsTensor() ) {
			ofLogError("ofxOnnx::runFloat") << "the output is not a tensor";
			return false;
		}
		const auto info = out[0].GetTensorTypeAndShapeInfo();
		if( info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ) {
			ofLogError("ofxOnnx::runFloat") << "the output is not float32";
			return false;
		}
		const float* data = out[0].GetTensorData<float>();
		output.assign(data, data + info.GetElementCount());
	} catch( const Ort::Exception& e ) {
		ofLogError("ofxOnnx::runFloat") << e.what();
		return false;
	}
	return true;
}

//--------------------------------------------------------------------
Ort::Session& ofxOnnx::getSession() {
	return mSession;
}

//--------------------------------------------------------------------
size_t ofxOnnx::getInputCount() {
	if (!mSession) return 0;
	return mSession.GetInputCount();
}

//--------------------------------------------------------------------
std::string ofxOnnx::getInputName(int aindex) {
	if (!mSession) return "";
	if( aindex < 0 || aindex >= getInputCount() ) {
		ofLogError("ofxOnnx::getInputName") << "index out of bounds: " << aindex << " / " << getInputCount();
	}
	Ort::AllocatorWithDefaultOptions alloc;
	auto name = mSession.GetInputNameAllocated(aindex, alloc);
	std::string rname = name.get();
	return rname;
}

//--------------------------------------------------------------------
ONNXTensorElementDataType ofxOnnx::getInputElementType(int aindex) {
	if(!mSession) {
		ofLogError("ofxOnnx::getInputShape") << "NO SESSION!";
		return {};
	}
	if( aindex < 0 || aindex >= getInputCount() ) {
		ofLogError("ofxOnnx::getInputShape") << "index out of bounds: " << aindex << " / " << getInputCount();
	}
	
	Ort::TypeInfo ti = mSession.GetInputTypeInfo(aindex);
	auto tt = ti.GetTensorTypeAndShapeInfo();
	
//	std::cout << "  " << aindex << ": " << getInputName(aindex) << "\n";
//	std::cout << "     dtype: " << DTypeName(tt.GetElementType()) << "\n";
	return tt.GetElementType();
}

//--------------------------------------------------------------------
std::vector<int64_t> ofxOnnx::getInputShape(int aindex) {
	if(!mSession)
		return {};
	if( aindex < 0 || aindex >= getInputCount() ) {
		ofLogError("ofxOnnx::getInputShape") << "index out of bounds: " << aindex << " / " << getInputCount();
	}
	Ort::TypeInfo ti = mSession.GetInputTypeInfo(aindex);
	auto tt = ti.GetTensorTypeAndShapeInfo();
	return tt.GetShape();
}

//--------------------------------------------------------------------
std::vector<const char*> ofxOnnx::getInputNames() {
	std::vector<const char*> rnames( mInputNames.size() );
	for( size_t i = 0; i < mInputNames.size(); i++ ) {
		rnames[i] = mInputNames[i].c_str();
	}
	return rnames;
}

//--------------------------------------------------------------------
size_t ofxOnnx::getOutputCount() {
	if (!mSession) return 0;
	return mSession.GetOutputCount();
}

//--------------------------------------------------------------------
std::string ofxOnnx::getOutputName(int aindex) {
	if (!mSession) return "";
	if (aindex < 0 || aindex >= getOutputCount()) {
		ofLogError("ofxOnnx::getOutputName") << "index out of bounds: " << aindex << " / " << getOutputCount();
	}
	Ort::AllocatorWithDefaultOptions alloc;
	auto name = mSession.GetOutputNameAllocated(aindex, alloc);
	std::string rname = name.get();
	return rname;
}

//--------------------------------------------------------------------
std::vector<int64_t> ofxOnnx::getOutputShape(int aindex) {
	if(!mSession)
		return {};
	if( aindex < 0 || aindex >= getOutputCount() ) {
		ofLogError("ofxOnnx::getOutputShape") << "index out of bounds: " << aindex << " / " << getOutputCount();
	}
	Ort::TypeInfo ti = mSession.GetOutputTypeInfo(aindex);
	auto tt = ti.GetTensorTypeAndShapeInfo();
	return tt.GetShape();
}

//--------------------------------------------------------------------
std::vector<const char*> ofxOnnx::getOutputNames() {
	std::vector<const char*> rnames( mOutputNames.size() );
	for( size_t i = 0; i < mOutputNames.size(); i++ ) {
		rnames[i] = mOutputNames[i].c_str();
	}
	return rnames;
}

//--------------------------------------------------------------------
void ofxOnnx::printMetaData() {
	if (!mSession) return;
	Ort::ModelMetadata md = mSession.GetModelMetadata();

	Ort::AllocatorWithDefaultOptions alloc;

	// ---- Model metadata (Allocated helpers) ----
	try {
		auto prod = md.GetProducerNameAllocated(alloc);
		std::cout << "Producer: " << prod.get() << "\n";
	} catch (...) { }
	try {
		auto graph = md.GetGraphNameAllocated(alloc);
		std::cout << "Graph name: " << graph.get() << "\n";
	} catch (...) { }
	try {
		auto desc = md.GetDescriptionAllocated(alloc);
		std::cout << "Description: " << desc.get() << "\n";
	} catch (...) { }
	try {
		auto domain = md.GetDomainAllocated(alloc);
		std::cout << "Domain: " << domain.get() << "\n";
	} catch (...) { }

	// Custom metadata map (keys only via Allocated helper, values by querying each)
	try {
		auto keys = md.GetCustomMetadataMapKeysAllocated(alloc);
		std::cout << "Custom metadata keys: ";
		for (size_t i = 0; i < keys.size(); ++i) {
			std::cout << (i ? ", " : "") << keys[i].get();
		}
		std::cout << "\n";
		// If you want values:
		// for (size_t i = 0; i < keys.size(); ++i) {
		//   auto val = md.LookupCustomMetadataMapAllocated(alloc, keys[i].get());
		//   std::cout << "  " << keys[i].get() << " = " << val.get() << "\n";
		// }
	} catch (...) { }
}

//--------------------------------------------------------------------
void ofxOnnx::printInputs() {
	if (!mSession) return;
	Ort::AllocatorWithDefaultOptions alloc;
	// ---- Inputs ----
	size_t n_in = mSession.GetInputCount();
	ofLogNotice("ofxOnnx") << mSettings.envName << " - Inputs (" << n_in << "):";
	for (size_t i = 0; i < n_in; ++i) {
		auto name = mSession.GetInputNameAllocated(i, alloc);
		Ort::TypeInfo ti = mSession.GetInputTypeInfo(i);
		auto tt = ti.GetTensorTypeAndShapeInfo();

		std::cout << "  " << i << ": " << name.get() << "\n";
		std::cout << "     dtype: " << DTypeName(tt.GetElementType()) << "\n";

		auto shape = tt.GetShape();
		std::cout << "     shape: [";
		for (size_t d = 0; d < shape.size(); ++d) {
			std::cout << (d ? ", " : "");
			if (shape[d] == -1)
				std::cout << "dynamic";
			else
				std::cout << shape[d];
		}
		std::cout << "]\n";
	}
}

//--------------------------------------------------------------------
void ofxOnnx::printOutputs() {
	if (!mSession) return;
	Ort::AllocatorWithDefaultOptions alloc;
	// ---- Outputs ----
	size_t n_out = mSession.GetOutputCount();
	ofLogNotice("ofxOnnx") << mSettings.envName << " - Outputs (" << n_out << "):";
	for (size_t i = 0; i < n_out; ++i) {
		auto name = mSession.GetOutputNameAllocated(i, alloc);
		Ort::TypeInfo ti = mSession.GetOutputTypeInfo(i);
		auto tt = ti.GetTensorTypeAndShapeInfo();

		std::cout << "  " << i << ": " << name.get() << "\n";
		std::cout << "     dtype: " << DTypeName(tt.GetElementType()) << "\n";

		auto shape = tt.GetShape();
		std::cout << "     shape: [";
		for (size_t d = 0; d < shape.size(); ++d) {
			std::cout << (d ? ", " : "");
			if (shape[d] == -1)
				std::cout << "dynamic";
			else
				std::cout << shape[d];
		}
		std::cout << "]\n";
	}
}
