#pragma once
#include "ofMain.h"
#include <cctype>
#include <string>
#include <vector>

// System prompt that makes the model answer with a GarmentCode design, in the shape of garmentgen-demo's design.json.
// A base model only follows this loosely; a model fine-tuned on the demo's data needs just the first sentence.
inline const std::string & garmentSystemPrompt() {
	static const std::string s =
		"You are a sewing pattern designer. Turn the garment description into a GarmentCode design. "
		"Reply with one JSON object and nothing else, no explanation and no code fence.\n"
		"Keys: \"v\" (always 1), \"upper\" (an upper-body garment class such as FittedShirt, or null), \"wb\" (waistband class such as StraightWB, or null), "
		"\"bottom\" (a skirt or pants class such as GodetSkirt, or null), and \"p\", an object of parameters named \"part.name\" "
		"(for example collar.f_collar, collar.b_collar, sleeve.sleeveless, pencil-skirt.length, pencil-skirt.flare). "
		"Lengths are fractions of the body height, flags are true or false.\n"
		"Example. Description: a knee-length sleeveless summer dress with a flared skirt and a V neckline. Reply:\n"
		"{\"v\":1,\"upper\":\"FittedShirt\",\"wb\":\"StraightWB\",\"bottom\":\"GodetSkirt\",\"p\":{\"waistband.waist\":1,\"waistband.width\":0.2,"
		"\"shirt.strapless\":false,\"collar.f_collar\":\"VNeckHalf\",\"collar.b_collar\":\"CircleNeckHalf\",\"collar.width\":0.2,\"collar.fc_depth\":0.4,"
		"\"collar.bc_depth\":0,\"sleeve.sleeveless\":true,\"sleeve.armhole_shape\":\"ArmholeCurve\",\"sleeve.connecting_width\":0.2,"
		"\"left.enable_asym\":false,\"godet-skirt.base\":\"PencilSkirt\",\"godet-skirt.insert_w\":11,\"godet-skirt.insert_depth\":14,"
		"\"godet-skirt.num_inserts\":6,\"godet-skirt.cuts_distance\":5,\"pencil-skirt.length\":0.33,\"pencil-skirt.flare\":1.335,"
		"\"pencil-skirt.low_angle\":0}}";
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
