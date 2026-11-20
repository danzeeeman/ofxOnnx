#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Byte-level BPE tokenizer for Phi-4 (the o200k-style tokenizer.json): reads vocab.json, merges.txt and
// added_tokens.json from the model folder. The pre-tokenizer regex is hand-matched; letter/number classes beyond
// Latin, Greek and Cyrillic are approximated (everything else that is not punctuation counts as a letter), which
// only matters for how unusual scripts are split, never for decoding.
class Tokenizer {
public:
	bool load(const std::string & absDir);

	// Special tokens ("<|user|>" ...) in the text are matched literally.
	std::vector<int64_t> encode(const std::string & text) const;
	// The raw bytes of one token ("" for special or unknown ids).
	const std::string & tokenBytes(int64_t id) const;
	std::string decode(const std::vector<int64_t> & ids) const;
	int64_t tokenId(const std::string & special) const;
	int64_t vocabSize() const { return (int64_t)idToBytes.size(); }

	// Length of the longest prefix of `bytes` that does not end inside a UTF-8 sequence.
	static size_t validUtf8Prefix(const std::string & bytes);

private:
	void encodeOrdinary(const std::string & text, std::vector<int64_t> & out) const;
	void bpe(const std::string & bytes, std::vector<int64_t> & out) const;

	std::unordered_map<std::string, int64_t> vocab; // byte-mapped token string -> id
	std::vector<std::string> idToBytes;             // id -> raw bytes
	std::unordered_map<std::string, int> mergeRank; // "a b" -> rank
	std::vector<std::pair<std::string, int64_t>> specials;
	std::string byteToUni[256];
	std::unordered_map<uint32_t, uint8_t> uniToByte;
};
