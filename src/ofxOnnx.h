
#pragma once
#include "ofMesh.h"
#include "ofLog.h"
#include "ofUtils.h"
#include <algorithm>
#include <array>
#include <string>
#include <thread>
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>

// wrapper for onnx runtime.
// https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages

// Platforms: macOS and Linux (CPU, CUDA on Linux) and Windows x64 (CPU). The Windows libraries are in
// libs/onnxruntime/lib/vs/x64 (onnxruntime.dll is copied next to the exe by the project generator).
//
// The GPU CUDA accelerated version on Linux is much faster.
// Download the binary here: https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4
// Must have CUDA installed.

// -- ONNX License -- //
// https://github.com/microsoft/onnxruntime?tab=MIT-1-ov-file
//MIT License
//
//Copyright (c) Microsoft Corporation
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
// ------------------ //


class ofxOnnx {
public:
	ofxOnnx() : mEnv(nullptr), mSession(nullptr), mAllocator(),
	mCpuMem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) { }

    struct Settings {
//        #if defined(OFX_ONNX_USE_CUDA)
		bool useCuda = false; // must have CUDA installed and linked.
		bool useCoreML = true; // only for macos
		int deviceId = 0;
		bool bPrintModelInfo = true;
//		#endif
		int numIntraOpThreads = std::max(1,(int)std::thread::hardware_concurrency()/2);
		of::filesystem::path modelPath;
		ofLogLevel logLevel = OF_LOG_NOTICE;
		std::string envName = "ofxOnnx";
	};

	bool load(ofxOnnx::Settings asettings);
	
	bool hasSession() {return (mSession ? true : false); }
	Ort::Session& getSession();
	Ort::MemoryInfo& getCpuMem() { return mCpuMem; }

	size_t getInputCount();
	std::string getInputName(int aindex);
	std::vector<int64_t> getInputShape(int aindex);
	ONNXTensorElementDataType getInputElementType(int aindex);
	std::vector<const char *> getInputNames();

	size_t getOutputCount();
	std::string getOutputName(int aindex);
	std::vector<int64_t> getOutputShape(int aindex);
	std::vector<const char*> getOutputNames();

	// One float vector in, one float vector out: an MLP policy, a classifier, a regressor. The input is given
	// the model's own input shape with every dynamic dimension set to 1, so a model taking [batch, n] is run with
	// batch 1. The result is copied into `output` (resized to fit). Meant for loops that run every frame or
	// faster: the name tables are built once at load, so nothing is rebuilt per call.
	// False (with a log line) if there is no session, the model does not have exactly one float input and one
	// float output, or the input has the wrong number of values.
	bool runFloat(const std::vector<float>& input, std::vector<float>& output);

	void printMetaData();
	void printInputs();
	void printOutputs();
	
	template<typename T>
	Ort::Value getInputCpuTensor(int aInputIndex, std::vector<T>& adata ) {
		auto shape = getInputShape(aInputIndex);
		for( auto& sv : shape ) {
			if( sv < 0 ) { sv = 1; }
		}
		return getInputCpuTensor( aInputIndex, adata, shape );
	}
	
	template<typename T>
	Ort::Value getInputCpuTensor(int aInputIndex, std::vector<T>& adata, std::vector<int64_t> aShape ) {
		auto shape = aShape;
		if( aShape.size() < 1) {
			shape = getInputShape(aInputIndex);
		}
		return Ort::Value::CreateTensor<T>(mCpuMem,
											   adata.data(), adata.size(),
											   shape.data(), shape.size()
											   );
	}
	
	template<typename T>
	Ort::Value getInputCpuTensor(std::vector<T>& adata, std::vector<int64_t> aShape ) {
		return Ort::Value::CreateTensor<T>(mCpuMem,
										   adata.data(), adata.size(),
										   aShape.data(), aShape.size()
										   );
	}
	
	
	Ort::Value getInputCpuTensorF(std::vector<float>& adata, std::vector<int64_t> aShape, ONNXTensorElementDataType atype ) {
		
		if(atype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ) {
			return createFloat16Tensor(adata, aShape );
		}
		
		return Ort::Value::CreateTensor<float>(mCpuMem,
											   adata.data(), adata.size(),
											   aShape.data(), aShape.size()
											   );
	}
	
	
	Ort::Value createFloatTensor(const std::vector<float>& adata,
								 const std::vector<int64_t>& ashape) {
		auto tensor = Ort::Value::CreateTensor<float>(mAllocator, ashape.data(), ashape.size());
		float* tensor_data = tensor. GetTensorMutableData<float>();
		std::copy(adata.begin(), adata.end(), tensor_data);
		return tensor;
	}
	
	Ort::Value createFloat16Tensor(const std::vector<float>& adata,
								   const std::vector<int64_t>& ashape) {
		// Allocate tensor with ONNX Runtime's memory
		auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(mAllocator, ashape.data(), ashape.size());
		
		// Get pointer to tensor's memory and fill it
		Ort::Float16_t* tensor_data = tensor.GetTensorMutableData<Ort::Float16_t>();
		for (size_t i = 0; i < adata.size(); ++i) {
			tensor_data[i] = Ort::Float16_t(adata[i]);
		}
		return tensor;
	}
	
	Ort::Value createInt64Tensor(const std::vector<int64_t>& adata,
								   const std::vector<int64_t>& ashape) {
		auto tensor = Ort::Value::CreateTensor<int64_t>(mAllocator, ashape.data(), ashape.size());
		int64_t* tensor_data = tensor. GetTensorMutableData<int64_t>();
		std::copy(adata.begin(), adata.end(), tensor_data);
		return tensor;
	}
	
	
	template <typename T>
	std::vector<Ort::Value> run( std::vector<T>& aInputData ) {
		return run( getInputCpuTensor( 0, aInputData ) );
	}
	
	template <typename T>
	std::vector<Ort::Value> run( std::vector<T>& aInputData, std::vector<int64_t> aShape ) {
		return run( getInputCpuTensor( 0, aInputData, aShape ) );
	}
	
	
	std::vector<Ort::Value> run( Ort::Value aTensor ) {
		if( !hasSession() ) {
			std::vector<Ort::Value> tmp;
			return tmp;
		}
		
		std::array<Ort::Value, 1> inputs{ std::move(aTensor) };  // uses move ctor
//		auto inNameStrs = getInputNames();
//		std::vector<const char *> inNames(inNameStrs.size());
//		for( size_t i = 0; i < inNameStrs.size(); i++ ) {
//			inNames[i] = inNameStrs[i].c_str();
//		}
//		auto outNameStrs = getOutputNames();
//		std::vector<const char *> outNames(outNameStrs.size());
//		for( size_t i = 0; i < outNameStrs.size(); i++ ) {
//			outNames[i] = outNameStrs[i].c_str();
//		}
		return mSession.Run(Ort::RunOptions { nullptr },
							getInputNames().data(), inputs.data(), getInputCount(),
							getOutputNames().data(), getOutputCount());
		
	}
	
	template <typename T>
	std::vector<Ort::Value> run( std::vector< std::vector<T> >& aInputData ) {
		
		if( !hasSession() ) {
			std::vector<Ort::Value> tmp;
			return tmp;
		}
		
		std::vector<Ort::Value> inputs(aInputData.size());
		for( size_t i = 0; i < inputs.size(); i++ ) {
			inputs[i] = getInputCpuTensor( i, aInputData[i] );
		}
		return mSession.Run(Ort::RunOptions { nullptr },
							getInputNames().data(), inputs.data(), getInputCount(),
							getOutputNames().data(), getOutputCount());
		
	}

protected:
	ofxOnnx::Settings mSettings;
	Ort::Env mEnv;
	// Ort::Session mSession;
//	std::unique_ptr<Ort::Session> mSession;
	Ort::Session mSession;

	Ort::MemoryInfo mCpuMem;
	Ort::AllocatorWithDefaultOptions mAllocator;
	
	// going to store them in a vector so we can
	// create std::vector<const char *> without issues.
	std::vector<std::string> mInputNames;
	std::vector<std::string> mOutputNames;
	// The same names as C strings, built once at load for runFloat().
	std::vector<const char*> mInputNamePtrs;
	std::vector<const char*> mOutputNamePtrs;
};
