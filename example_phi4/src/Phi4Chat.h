#pragma once
#include "Tokenizer.h"
#include "ofxOnnx.h"
#include <atomic>
#include <functional>
#include <random>

// Chat with Phi-4 through ofxOnnx: the tokenizer, the chat template, a KV-cache decode loop whose cache stays on the
// GPU (Ort::IoBinding), and a sampler. onnxruntime-genai is not used.
class Phi4Chat {
public:
	struct Options {
		std::string modelDir = "Phi-4-mini-reasoning-onnx/gpu/gpu-int4-rtn-block-32"; // relative to bin/data
		bool useCuda = true;
		std::string cudaDllPath;
		std::string system = "Your name is Phi, an AI math expert developed by Microsoft.";
		int maxNewTokens = 1536;
		int maxContext = 8192;
		float temperature = 0.8f; // <= 0: greedy
		float topP = 0.95f;
		int topK = 40;
	};

	bool load(const Options & o);
	bool isGpu() const { return gpu; }

	// Answers `user` in the running conversation; blocks, calling onToken with each piece of text as it is decoded.
	// Run it on a thread; cancel() stops it from another.
	void chat(const std::string & user, const std::function<void(const std::string &)> & onToken);
	void cancel() { cancelFlag = true; }
	bool cancelled() const { return cancelFlag; }
	void reset(); // new conversation (not while chat() is running)

	float tokensPerSecond() const { return tps; }
	int contextUsed() const { return past; }
	int contextMax() const { return opt.maxContext; }
	int promptTokens() const { return lastPrompt; }
	Options & options() { return opt; }

private:
	std::vector<float> step(const std::vector<int64_t> & ids);
	int64_t sample(std::vector<float> & logits);

	Options opt;
	Tokenizer tok;
	ofxOnnx onnx;
	// phi4mm layout (onnxruntime-genai): phi-4-mm-embedding.onnx turns token ids into inputs_embeds, and the text model
	// takes those plus LoRA weight inputs, which are zero-filled here (the base model) until a text adapter is loaded.
	bool mm = false;
	ofxOnnx embed;
	std::vector<Ort::Value> lora;
	std::vector<std::string> loraNames;
	bool gpu = false;
	int layers = 32;
	std::vector<int64_t> kvShape; // [1, heads, 0, head_size]
	ONNXTensorElementDataType kvType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
	ONNXTensorElementDataType idType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
	bool hasPositionIds = true;
	std::vector<std::string> pastNames, presentNames;
	std::vector<Ort::Value> kv;
	int past = 0;
	std::vector<int64_t> pending; // tokens decided but not yet in the cache
	std::vector<int64_t> eosIds;
	int64_t endId = 200020;
	std::atomic<bool> cancelFlag { false };
	std::atomic<float> tps { 0 };
	std::atomic<int> lastPrompt { 0 };
	std::mt19937 rng { std::random_device {}() };
};
