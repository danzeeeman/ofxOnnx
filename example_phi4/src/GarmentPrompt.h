#pragma once
#include "ofMain.h"
#include <cctype>
#include <string>
#include <vector>

// System prompt that makes the model answer with a GarmentCode design, in the shape of garmentgen-demo's design.json.
// A base model only follows this loosely; a model fine-tuned on the demo's data needs just the first sentence.
// The prompt the garmentgen model was fine-tuned on (garmentgen/model/phi4_data.py SYSTEM_DESIGN, and the first "text" template of
// configs/instructions.yaml): no schema and no examples, the model already knows the design format.
inline const std::string & garmentSystemPrompt() {
	static const std::string s = "You are a sewing pattern-making assistant. Reply with one GarmentDesign JSON v1 object and nothing else.";
	return s;
}
inline std::string garmentInstruction(const std::string & description) {
	const char * ws = " \t\r\n";
	const size_t b = description.find_first_not_of(ws), e = description.find_last_not_of(ws);
	return "Create a sewing pattern for: " + (b == std::string::npos ? std::string() : description.substr(b, e - b + 1));
}

// For a model that was not fine-tuned on designs: a system prompt that explains the format, with ranges and examples.
inline const std::string & baseModelSystemPrompt() {
	static const std::string s = R"PROMPT(You are a sewing pattern designer. Turn the garment description into a GarmentCode design: one JSON object and nothing else, no explanation, no code fence.

"v" is always 1.
"upper" is the top: "FittedShirt" (close fitting), "Shirt" (loose), "Jacket", "FittedJacket", or null for no top.
"wb" is the waistband: "StraightWB", "FittedWB", or null.
"bottom" is a skirt or trousers: "SkirtCircle" (circle or full skirt), "AsymmSkirtCircle", "GodetSkirt", "Pants" (trousers and shorts), "Skirt2" (straight or A-line), "SkirtManyPanels", "PencilSkirt", "SkirtLevels" (tiered), or null.
A dress is a top plus a skirt. Trousers, jeans and shorts are Pants with no top.
"p" holds the parameters of the parts you chose, named "part.name". Lengths are fractions of the body, larger is longer or wider.
  shirt.length 0.5-3.5 (1.2 is hip length, 2 or more is a tunic or dress), shirt.width 1-1.3, shirt.flare 0.7-1.6, shirt.strapless true/false
  sleeve.sleeveless true/false (false gives sleeves), sleeve.length 0.1-1.15 (0.3 short, 1 long), sleeve.end_width 0.2-2
  collar.f_collar and collar.b_collar (front and back neckline): CircleNeckHalf, CurvyNeckHalf, VNeckHalf, SquareNeckHalf, TrapezoidNeckHalf
  collar.fc_depth 0.3-2 and collar.bc_depth 0-2 (deeper neckline), collar.width -0.5-1
  waistband.waist 1-2, waistband.width 0.1-1
  pencil-skirt.length 0.2-0.95, pencil-skirt.flare 0.6-1.5, pencil-skirt.front_slit, back_slit, left_slit, right_slit 0-0.9
  skirt.length -0.2-0.95 (Skirt2), flare-skirt.length -0.2-0.95, flare-skirt.suns 0.1-1.95 (fullness of a circle skirt)
  pants.length 0.2-0.9, pants.width 1-1.5, pants.flare 0.5-1.2, pants.rise 0.5-1
Set every value to match the description. Different descriptions give different designs; do not repeat an example unless the description matches it.

Examples.
Description: a short sleeve fitted top with a V neck
{"v":1,"upper":"FittedShirt","wb":null,"bottom":null,"p":{"shirt.strapless":false,"sleeve.sleeveless":false,"sleeve.length":0.3,"collar.f_collar":"VNeckHalf","collar.fc_depth":0.8}}
Description: wide leg trousers with a high waistband
{"v":1,"upper":null,"wb":"StraightWB","bottom":"Pants","p":{"waistband.waist":1,"waistband.width":0.6,"pants.length":0.9,"pants.width":1.4,"pants.flare":1.1,"pants.rise":1}}
Description: a long full circle skirt
{"v":1,"upper":null,"wb":"FittedWB","bottom":"SkirtCircle","p":{"waistband.waist":1,"waistband.width":0.3,"flare-skirt.length":0.8,"flare-skirt.suns":1.5,"flare-skirt.rise":1}})PROMPT";
	return s;
}

// Turns a model answer into a valid design JSON string, or "" if it cannot. Looks after any <think>...</think>, starts at
// the first '{', and repairs what models get wrong: trailing commas, an answer cut off mid-object (open strings and
// brackets are closed) and top-level keys left out (v = 1, upper / wb / bottom = null, p = {}). `note` says what was
// done, or why it failed.
inline std::string extractDesignJson(const std::string & text, std::string * note = nullptr) {
	using json = nlohmann::ordered_json;
	auto say = [&](const std::string & m) { if (note) *note = m; };
	size_t from = text.rfind("</think>");
	from = from == std::string::npos ? 0 : from + 8;
	size_t a = text.find('{', from);
	if (a == std::string::npos) { say("no JSON object in the answer"); return ""; }

	// the object: up to its closing brace, or to the end if it never closes
	std::string body;
	std::vector<char> stack;
	bool str = false, closed = false;
	for (size_t i = a; i < text.size() && !closed; i++) {
		char c = text[i];
		body += c;
		if (str) {
			if (c == '\\' && i + 1 < text.size()) body += text[++i];
			else if (c == '"') str = false;
		} else if (c == '"') str = true;
		else if (c == '{' || c == '[') stack.push_back(c == '{' ? '}' : ']');
		else if ((c == '}' || c == ']') && !stack.empty()) { stack.pop_back(); closed = stack.empty(); }
	}
	std::string fixes;
	if (!closed) {
		if (str) body += '"';
		while (!body.empty() && (body.back() == ',' || body.back() == ':' || std::isspace((unsigned char)body.back()))) body.pop_back();
		for (size_t k = stack.size(); k-- > 0;) body += stack[k];
		fixes += "closed an unfinished object; ";
	}
	// trailing commas before } or ]
	std::string clean;
	str = false;
	for (size_t i = 0; i < body.size(); i++) {
		char c = body[i];
		if (str) { clean += c; if (c == '\\' && i + 1 < body.size()) clean += body[++i]; else if (c == '"') str = false; continue; }
		if (c == '"') str = true;
		if (c == ',') {
			size_t k = i + 1;
			while (k < body.size() && std::isspace((unsigned char)body[k])) k++;
			if (k < body.size() && (body[k] == '}' || body[k] == ']')) { fixes += "removed a trailing comma; "; continue; }
		}
		clean += c;
	}
	json j = json::parse(clean, nullptr, false);
	if (j.is_discarded() || !j.is_object()) { say("the answer is not valid JSON"); return ""; }

	if (!j.contains("v") || !j["v"].is_number()) { j["v"] = 1; fixes += "set v = 1; "; }
	for (const char * k : { "upper", "wb", "bottom" })
		if (!j.contains(k) || !(j[k].is_string() || j[k].is_null())) { j[k] = nullptr; fixes += std::string("set ") + k + " = null; "; }
	if (!j.contains("p") || !j["p"].is_object()) { j["p"] = json::object(); fixes += "set p = {}; "; }
	// canonical order: v, upper, wb, bottom, p
	json out = json::object();
	for (const char * k : { "v", "upper", "wb", "bottom", "p" }) out[k] = j[k];
	say(fixes);
	return out.dump(1);
}
