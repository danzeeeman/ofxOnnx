#include "Tokenizer.h"
#include "ofMain.h"
#include <fstream>
#include <sstream>

namespace {
enum Cls { kUpper, kLower, kOtherLetter, kMark, kNumber, kSpace, kOther };

Cls classify(uint32_t c) {
	if (c < 0x80) {
		if (c >= 'A' && c <= 'Z') return kUpper;
		if (c >= 'a' && c <= 'z') return kLower;
		if (c >= '0' && c <= '9') return kNumber;
		if (c == ' ' || (c >= 9 && c <= 13)) return kSpace;
		return kOther;
	}
	if (c < 0x100) {
		if (c == 0x85 || c == 0xA0) return kSpace;
		if (c == 0xAA || c == 0xBA) return kOtherLetter;
		if (c == 0xB5) return kLower;
		if (c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE)) return kNumber;
		if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return kUpper;
		if (c >= 0xDF && c != 0xF7) return kLower;
		return kOther;
	}
	if (c < 0x180) return (c & 1) ? kLower : kUpper;
	if (c < 0x250) return kOtherLetter;
	if (c >= 0x300 && c <= 0x36F) return kMark;
	if (c >= 0x370 && c <= 0x3FF) {
		if (c >= 0x391 && c <= 0x3A9) return kUpper;
		if (c >= 0x3B1 && c <= 0x3C9) return kLower;
		return kOtherLetter;
	}
	if (c >= 0x400 && c <= 0x42F) return kUpper;
	if (c >= 0x430 && c <= 0x45F) return kLower;
	if (c >= 0x460 && c <= 0x52F) return (c & 1) ? kLower : kUpper;
	if ((c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) return kSpace;
	if (c >= 0x2000 && c <= 0x2BFF) return kOther;
	if (c >= 0x3000 && c <= 0x303F) return kOther;
	if (c >= 0xFE00 && c <= 0xFE0F) return kMark;
	if (c >= 0xFF10 && c <= 0xFF19) return kNumber;
	if ((c >= 0xFF01 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20)) return kOther;
	if (c >= 0xE000 && c <= 0xF8FF) return kOther;
	if (c >= 0x1F000 && c <= 0x1FAFF) return kOther;
	return kOtherLetter;
}

void appendUtf8(std::string & s, uint32_t c) {
	if (c < 0x80) s += (char)c;
	else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
	else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
	else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}

// Decode UTF-8 into code points; off[i] is the byte offset of code point i, off.back() the length.
void decodeUtf8(const std::string & s, std::vector<uint32_t> & cps, std::vector<size_t> & off) {
	size_t i = 0;
	while (i < s.size()) {
		unsigned char b = (unsigned char)s[i];
		int len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
		uint32_t c = 0xFFFD;
		if (len == 0 || i + len > s.size()) len = 1;
		else if (len == 1) c = b;
		else {
			c = b & (0xFF >> (len + 1));
			bool ok = true;
			for (int k = 1; k < len; k++) {
				unsigned char cb = (unsigned char)s[i + k];
				if ((cb & 0xC0) != 0x80) { ok = false; break; }
				c = (c << 6) | (cb & 0x3F);
			}
			if (!ok) { c = 0xFFFD; len = 1; }
		}
		cps.push_back(c);
		off.push_back(i);
		i += len;
	}
	off.push_back(s.size());
}

struct PreTok {
	std::vector<uint32_t> cps;
	std::vector<Cls> cl;
	size_t n;

	bool isL(size_t i) const { return cl[i] == kUpper || cl[i] == kLower || cl[i] == kOtherLetter; }
	bool isA(size_t i) const { return cl[i] != kLower && (isL(i) || cl[i] == kMark); } // Lu Lt Lm Lo M
	bool isB(size_t i) const { return cl[i] != kUpper && (isL(i) || cl[i] == kMark); } // Ll Lm Lo M
	bool isNL(size_t i) const { return cps[i] == '\n' || cps[i] == '\r'; }
	bool prefixOK(size_t i) const { return !isNL(i) && !isL(i) && cl[i] != kNumber; }

	size_t contraction(size_t i) const {
		if (i >= n || cps[i] != '\'') return 0;
		auto lower = [&](size_t k) { return k < n ? (cps[k] | 0x20) : 0u; };
		uint32_t a = lower(i + 1);
		if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
		uint32_t b = lower(i + 2);
		if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
		return 0;
	}

	// [^\r\n\p{L}\p{N}]?[Lu Lt Lm Lo M]*[Ll Lm Lo M]+ contraction?
	size_t alt1(size_t i) const {
		for (int pre = 1; pre >= 0; pre--) {
			size_t p = i;
			if (pre) { if (!prefixOK(p)) continue; p++; }
			size_t a = p;
			while (a < n && isA(a)) a++;
			for (size_t k = a + 1; k-- > p;) {
				size_t q = k;
				while (q < n && isB(q)) q++;
				if (q > k) return q + contraction(q);
			}
		}
		return 0;
	}
	// [^\r\n\p{L}\p{N}]?[Lu Lt Lm Lo M]+[Ll Lm Lo M]* contraction?
	size_t alt2(size_t i) const {
		for (int pre = 1; pre >= 0; pre--) {
			size_t p = i;
			if (pre) { if (!prefixOK(p)) continue; p++; }
			size_t a = p;
			while (a < n && isA(a)) a++;
			if (a > p) {
				size_t q = a;
				while (q < n && isB(q)) q++;
				return q + contraction(q);
			}
		}
		return 0;
	}

	// end of the token starting at i
	size_t next(size_t i) const {
		if (size_t e = alt1(i)) return e;
		if (size_t e = alt2(i)) return e;
		if (cl[i] == kNumber) { size_t e = i; while (e < n && e - i < 3 && cl[e] == kNumber) e++; return e; }
		{ // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
			size_t s = i;
			if (cps[s] == ' ') s++;
			size_t e = s;
			while (e < n && cl[e] != kSpace && !isL(e) && cl[e] != kNumber) e++;
			if (e > s) {
				while (e < n && (isNL(e) || cps[e] == '/')) e++;
				return e;
			}
		}
		// whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
		size_t run = i;
		while (run < n && cl[run] == kSpace) run++;
		size_t lastNL = n;
		for (size_t k = i; k < run; k++) if (isNL(k)) lastNL = k;
		if (lastNL != n) return lastNL + 1;
		if (run == n) return run;
		if (run - i > 1) return run - 1;
		return run;
	}
};
} // namespace

bool Tokenizer::load(const std::string & absDir) {
	// GPT-2's byte <-> printable unicode table
	std::vector<int> bs;
	for (int b = 33; b <= 126; b++) bs.push_back(b);
	for (int b = 161; b <= 172; b++) bs.push_back(b);
	for (int b = 174; b <= 255; b++) bs.push_back(b);
	bool in[256] = {};
	for (int b : bs) in[b] = true;
	int extra = 0;
	for (int b = 0; b < 256; b++) {
		uint32_t cp = in[b] ? b : 256 + extra++;
		byteToUni[b].clear();
		appendUtf8(byteToUni[b], cp);
		uniToByte[cp] = (uint8_t)b;
	}

	auto vocabFile = ofBufferFromFile(absDir + "/vocab.json");
	if (vocabFile.size() == 0) { ofLogError("Tokenizer") << "no vocab.json in " << absDir; return false; }
	ofJson j = ofJson::parse(vocabFile.getText(), nullptr, false);
	if (j.is_discarded()) { ofLogError("Tokenizer") << "vocab.json is not valid JSON"; return false; }
	int64_t maxId = 0;
	for (auto it = j.begin(); it != j.end(); ++it) {
		int64_t id = it.value().get<int64_t>();
		vocab[it.key()] = id;
		maxId = std::max(maxId, id);
	}

	std::ifstream mf(absDir + "/merges.txt", std::ios::binary);
	if (!mf) { ofLogError("Tokenizer") << "no merges.txt in " << absDir; return false; }
	std::string line;
	int rank = 0;
	while (std::getline(mf, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty() || line[0] == '#') continue;
		mergeRank.emplace(line, rank++);
	}

	auto addedFile = ofBufferFromFile(absDir + "/added_tokens.json");
	if (addedFile.size() > 0) {
		ofJson a = ofJson::parse(addedFile.getText(), nullptr, false);
		if (!a.is_discarded())
			for (auto it = a.begin(); it != a.end(); ++it) {
				specials.emplace_back(it.key(), it.value().get<int64_t>());
				maxId = std::max(maxId, it.value().get<int64_t>());
			}
	}
	specials.emplace_back("<|endoftext|>", 199999);
	maxId = std::max<int64_t>(maxId, 199999);

	idToBytes.assign(maxId + 1, "");
	for (auto & kv : vocab) {
		std::vector<uint32_t> cps;
		std::vector<size_t> off;
		decodeUtf8(kv.first, cps, off);
		std::string raw;
		for (uint32_t c : cps) {
			auto f = uniToByte.find(c);
			if (f != uniToByte.end()) raw += (char)f->second;
		}
		if (kv.second >= 0 && kv.second <= maxId) idToBytes[kv.second] = raw;
	}
	return true;
}

int64_t Tokenizer::tokenId(const std::string & special) const {
	for (auto & s : specials) if (s.first == special) return s.second;
	return -1;
}

const std::string & Tokenizer::tokenBytes(int64_t id) const {
	static const std::string empty;
	if (id < 0 || id >= (int64_t)idToBytes.size()) return empty;
	for (auto & s : specials) if (s.second == id) return empty;
	return idToBytes[id];
}

std::string Tokenizer::decode(const std::vector<int64_t> & ids) const {
	std::string out;
	for (auto id : ids) out += tokenBytes(id);
	return out;
}

size_t Tokenizer::validUtf8Prefix(const std::string & b) {
	size_t n = b.size();
	for (size_t back = 1; back <= 3 && back <= n; back++) {
		unsigned char c = (unsigned char)b[n - back];
		if ((c & 0xC0) == 0x80) continue; // continuation byte, keep looking for the lead
		int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
		return (size_t)len > back ? n - back : n;
	}
	return n;
}

void Tokenizer::bpe(const std::string & bytes, std::vector<int64_t> & out) const {
	std::vector<std::string> sym;
	for (unsigned char b : bytes) sym.push_back(byteToUni[b]);
	while (sym.size() > 1) {
		int best = INT_MAX;
		size_t at = 0;
		for (size_t i = 0; i + 1 < sym.size(); i++) {
			auto f = mergeRank.find(sym[i] + " " + sym[i + 1]);
			if (f != mergeRank.end() && f->second < best) { best = f->second; at = i; }
		}
		if (best == INT_MAX) break;
		sym[at] += sym[at + 1];
		sym.erase(sym.begin() + at + 1);
	}
	for (auto & s : sym) {
		auto f = vocab.find(s);
		if (f != vocab.end()) out.push_back(f->second);
	}
}

void Tokenizer::encodeOrdinary(const std::string & text, std::vector<int64_t> & out) const {
	PreTok p;
	std::vector<size_t> off;
	decodeUtf8(text, p.cps, off);
	p.n = p.cps.size();
	for (uint32_t c : p.cps) p.cl.push_back(classify(c));
	size_t i = 0;
	while (i < p.n) {
		size_t e = p.next(i);
		if (e <= i) e = i + 1;
		bpe(text.substr(off[i], off[e] - off[i]), out);
		i = e;
	}
}

std::vector<int64_t> Tokenizer::encode(const std::string & text) const {
	std::vector<int64_t> out;
	size_t pos = 0;
	while (pos < text.size()) {
		size_t best = std::string::npos, bestLen = 0;
		int64_t bestId = 0;
		for (auto & s : specials) {
			size_t f = text.find(s.first, pos);
			if (f != std::string::npos && (f < best || (f == best && s.first.size() > bestLen))) {
				best = f; bestLen = s.first.size(); bestId = s.second;
			}
		}
		if (best == std::string::npos) { encodeOrdinary(text.substr(pos), out); break; }
		if (best > pos) encodeOrdinary(text.substr(pos, best - pos), out);
		out.push_back(bestId);
		pos = best + bestLen;
	}
	return out;
}
