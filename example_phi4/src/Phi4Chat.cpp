#include "Phi4Chat.h"
#include "ofMain.h"
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>

bool Phi4Chat::load(const Options & o) {
	opt = o;
	const std::string dir = ofToDataPath(o.modelDir, true);
	if (!tok.load(dir)) return false;
	endId = tok.tokenId("<|end|>");

	eosIds = { 200020, 199999 };
	auto cfg = ofBufferFromFile(dir + "/genai_config.json");
	if (cfg.size() > 0) {
		ofJson j = ofJson::parse(cfg.getText(), nullptr, false);
		if (!j.is_discarded() && j.contains("model") && j["model"].contains("eos_token_id")) {
			auto & e = j["model"]["eos_token_id"];
			eosIds.clear();
			if (e.is_array()) for (auto & v : e) eosIds.push_back(v.get<int64_t>());
			else eosIds.push_back(e.get<int64_t>());
		}
		if (!j.is_discarded() && j.contains("model") && j["model"].contains("decoder"))
			layers = j["model"]["decoder"].value("num_hidden_layers", 32);
	}

	ofxOnnx::Settings s;
	mm = ofFile::doesFileExist(dir + "/phi-4-mm-text.onnx");
	s.modelPath = o.modelDir + (mm ? "/phi-4-mm-text.onnx" : "/model.onnx");
	s.useCuda = o.useCuda;
	s.cudaDllPath = o.cudaDllPath;
	s.bPrintModelInfo = false;
	s.parallel = false;
	s.logLevel = OF_LOG_WARNING;
	if (!onnx.load(s)) return false;
	gpu = onnx.getSettings().useCuda;
	if (o.useCuda && !gpu) ofLogWarning("Phi4Chat") << "CUDA was asked for but this session is on the CPU";

	hasPositionIds = false;
	bool foundPast = false;
	for (size_t i = 0; i < onnx.getInputCount(); i++) {
		std::string n = onnx.getInputName((int)i);
		if (n == "position_ids") hasPositionIds = true;
		if (n == "input_ids") idType = onnx.getInputElementType((int)i);
		if (n == "past_key_values.0.key") {
			kvType = onnx.getInputElementType((int)i);
			kvShape = onnx.getInputShape((int)i);
			foundPast = true;
		}
	}
	if (!foundPast || kvShape.size() != 4) {
		ofLogError("Phi4Chat") << "this model has no past_key_values.N.key inputs of rank 4: not a KV-cache decoder";
		return false;
	}
	kvShape[0] = 1;
	kvShape[2] = 0;
	for (int i = 0; i < layers; i++) {
		for (const char * kind : { "key", "value" }) {
			pastNames.push_back("past_key_values." + std::to_string(i) + "." + kind);
			presentNames.push_back("present." + std::to_string(i) + "." + kind);
		}
	}
	if (mm) {
		ofxOnnx::Settings es;
		es.modelPath = o.modelDir + "/phi-4-mm-embedding.onnx";
		es.useCuda = false; // a table lookup; the CPU is enough and keeps the GPU for the decoder
		es.bPrintModelInfo = false;
		es.parallel = false;
		es.logLevel = OF_LOG_WARNING;
		if (!embed.load(es)) return false;
			// LoRA weights the text model asks for: all zero (scales 0), shapes from the model with the free dimensions
		// set to the rank (16) for lora_A's rows and one block for lora_B's input.
		lora.clear();
		loraNames.clear();
		std::map<std::string, std::vector<int64_t>> q4;
		for (size_t i = 0; i < onnx.getInputCount(); i++) {
			std::string n = onnx.getInputName((int)i);
			if (n.find("lora_") == std::string::npos) continue;
			auto shape = onnx.getInputShape((int)i);
			bool isA = n.find("lora_A") != std::string::npos;
			if (n.size() > 9 && n.compare(n.size() - 9, 9, "weight_Q4") == 0) {
				for (size_t d = 0; d < shape.size(); d++)
					if (shape[d] < 0) shape[d] = (isA && d == 0) ? 16 : 1;
				q4[n.substr(0, n.size() - 9)] = shape;
			}
		}
		Ort::AllocatorWithDefaultOptions alloc;
		for (size_t i = 0; i < onnx.getInputCount(); i++) {
			std::string n = onnx.getInputName((int)i);
			if (n.find("lora_") == std::string::npos) continue;
			std::vector<int64_t> shape;
			auto type = onnx.getInputElementType((int)i);
			if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) shape = q4[n.substr(0, n.size() - 9)];
			else {
				auto it = q4.find(n.substr(0, n.size() - 13)); // "weight_scales": one scale per block of the matching weight_Q4
				if (it == q4.end()) { ofLogError("Phi4Chat") << "no weight_Q4 for " << n; return false; }
				shape = { it->second[0] * it->second[1] };
			}
			Ort::Value v = Ort::Value::CreateTensor(alloc, shape.data(), shape.size(), type);
			size_t bytes = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : 2, count = 1;
			for (auto d : shape) count *= d;
			memset(v.GetTensorMutableRawData(), 0, count * bytes);
			loraNames.push_back(n);
			lora.push_back(std::move(v));
		}
	}
	reset();
	return true;
}

void Phi4Chat::reset() {
	kv.clear();
	past = 0;
	pending.clear();
	tps = 0;
	lastPrompt = 0;
}

std::vector<float> Phi4Chat::step(const std::vector<int64_t> & ids) {
	Ort::Session & session = onnx.getSession();
	const int64_t n = (int64_t)ids.size();
	const int64_t total = past + n;

	if (kv.empty()) {
		Ort::AllocatorWithDefaultOptions alloc;
		for (size_t i = 0; i < pastNames.size(); i++)
			kv.push_back(Ort::Value::CreateTensor(alloc, kvShape.data(), kvShape.size(), kvType));
	}

	std::vector<int64_t> idsStore(ids), mask(total, 1), pos(n);
	std::iota(pos.begin(), pos.end(), (int64_t)past);
	std::vector<int32_t> ids32(ids.begin(), ids.end()), mask32(mask.begin(), mask.end()), pos32(pos.begin(), pos.end());
	const bool i32 = idType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	auto mk = [&](std::vector<int64_t> & v64, std::vector<int32_t> & v32, int64_t len) {
		std::array<int64_t, 2> shape { 1, len };
		return i32 ? Ort::Value::CreateTensor<int32_t>(onnx.getCpuMem(), v32.data(), v32.size(), shape.data(), 2)
				   : Ort::Value::CreateTensor<int64_t>(onnx.getCpuMem(), v64.data(), v64.size(), shape.data(), 2);
	};
	Ort::Value inIds = mk(idsStore, ids32, n);
	Ort::Value inEmb { nullptr };
	if (mm) {
		// token ids -> embeddings (no images or audio: empty feature tensors)
		std::array<int64_t, 2> fs { 0, 3072 };
		Ort::AllocatorWithDefaultOptions alloc;
		Ort::Value img = Ort::Value::CreateTensor(alloc, fs.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
		Ort::Value aud = Ort::Value::CreateTensor(alloc, fs.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
		std::vector<Ort::Value> in;
		in.push_back(std::move(inIds));
		in.push_back(std::move(img));
		in.push_back(std::move(aud));
		const char * inNames[] = { "input_ids", "image_features", "audio_features" };
		const char * outNames[] = { "inputs_embeds" };
		auto out = embed.getSession().Run(Ort::RunOptions { nullptr }, inNames, in.data(), 3, outNames, 1);
		inEmb = std::move(out[0]);
	}
	Ort::Value inMask = mk(mask, mask32, total);
	Ort::Value inPos = mk(pos, pos32, n);

	Ort::IoBinding binding(session);
	if (mm) {
		binding.BindInput("inputs_embeds", inEmb);
		for (size_t i = 0; i < lora.size(); i++) binding.BindInput(loraNames[i].c_str(), lora[i]);
	} else {
		binding.BindInput("input_ids", inIds);
	}
	binding.BindInput("attention_mask", inMask);
	if (hasPositionIds) binding.BindInput("position_ids", inPos);
	for (size_t i = 0; i < pastNames.size(); i++) binding.BindInput(pastNames[i].c_str(), kv[i]);

	Ort::MemoryInfo dev = gpu ? Ort::MemoryInfo("Cuda", OrtDeviceAllocator, onnx.getSettings().deviceId, OrtMemTypeDefault)
							  : Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	binding.BindOutput("logits", onnx.getCpuMem());
	for (auto & name : presentNames) binding.BindOutput(name.c_str(), dev);

	session.Run(Ort::RunOptions { nullptr }, binding);
	std::vector<Ort::Value> outs = binding.GetOutputValues();

	std::vector<float> last;
	{
		auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
		const int64_t vocab = shape.back(), rows = shape[shape.size() - 2];
		last.resize(vocab);
		if (outs[0].GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
			const Ort::Float16_t * p = outs[0].GetTensorData<Ort::Float16_t>() + (rows - 1) * vocab;
			for (int64_t i = 0; i < vocab; i++) last[i] = p[i].ToFloat();
		} else {
			const float * p = outs[0].GetTensorData<float>() + (rows - 1) * vocab;
			std::copy(p, p + vocab, last.begin());
		}
	}
	kv.clear();
	for (size_t i = 1; i < outs.size(); i++) kv.push_back(std::move(outs[i]));
	past += (int)n;
	return last;
}

int64_t Phi4Chat::sample(std::vector<float> & l) {
	const float neg = -std::numeric_limits<float>::infinity();
	for (size_t i = (size_t)tok.vocabSize(); i < l.size(); i++) l[i] = neg; // padding ids
	if (opt.temperature <= 0.f) return std::max_element(l.begin(), l.end()) - l.begin();

	const size_t k = std::min<size_t>(std::max(1, opt.topK), l.size());
	std::vector<int64_t> idx(l.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t a, int64_t b) { return l[a] > l[b]; });
	std::vector<double> p(k);
	double sum = 0, top = l[idx[0]];
	for (size_t i = 0; i < k; i++) { p[i] = std::exp((l[idx[i]] - top) / opt.temperature); sum += p[i]; }
	double cum = 0;
	size_t keep = k;
	for (size_t i = 0; i < k; i++) {
		p[i] /= sum;
		cum += p[i];
		if (cum >= opt.topP) { keep = i + 1; break; }
	}
	std::discrete_distribution<size_t> d(p.begin(), p.begin() + keep);
	return idx[d(rng)];
}

void Phi4Chat::chat(const std::string & user, const std::function<void(const std::string &)> & onToken) {
	cancelFlag = false;
	try {
		std::vector<int64_t> in = pending;
		pending.clear();
		std::string text;
		if (past == 0 && in.empty()) text += "<|system|>" + opt.system + "<|end|>";
		text += "<|user|>" + user + "<|end|><|assistant|>";
		auto enc = tok.encode(text);
		in.insert(in.end(), enc.begin(), enc.end());
		lastPrompt = (int)in.size();
		if (past + (int)in.size() + 16 > opt.maxContext) {
			pending = {};
			onToken("[The context window is full. Press F1 to start a new chat.]");
			return;
		}

		std::vector<float> logits;
		const size_t chunk = 256;
		for (size_t off = 0; off < in.size(); off += chunk) {
			std::vector<int64_t> part(in.begin() + off, in.begin() + std::min(in.size(), off + chunk));
			logits = step(part);
		}

		using clock = std::chrono::steady_clock;
		auto t0 = clock::now();
		int generated = 0;
		std::string bytes;
		auto flush = [&](bool all) {
			size_t n = all ? bytes.size() : Tokenizer::validUtf8Prefix(bytes);
			if (n > 0) { onToken(bytes.substr(0, n)); bytes.erase(0, n); }
		};
		while (true) {
			int64_t t = sample(logits);
			if (std::find(eosIds.begin(), eosIds.end(), t) != eosIds.end()) { pending = { endId }; break; }
			bytes += tok.tokenBytes(t);
			flush(false);
			generated++;
			double secs = std::chrono::duration<double>(clock::now() - t0).count();
			if (secs > 0.05) tps = (float)(generated / secs);
			if (cancelFlag || generated >= opt.maxNewTokens || past + 1 >= opt.maxContext) { pending = { t, endId }; break; }
			logits = step({ t });
		}
		flush(true);
	} catch (const Ort::Exception & e) {
		onToken(std::string("\n[onnxruntime error: ") + e.what() + "]");
		reset();
	}
}
