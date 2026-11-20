#include "ofApp.h"
#include "GarmentPrompt.h"
#include <fstream>
#include <iostream>

namespace {
const ofColor kBg(14, 16, 20), kPanel(23, 26, 33), kPanel2(30, 34, 43), kBorder(42, 47, 59);
const ofColor kText(230, 232, 238), kMuted(138, 145, 163), kAccent(76, 125, 255), kUser(47, 91, 216);
const ofColor kGreen(61, 220, 151), kAmber(255, 180, 84);
const float kHeader = 64, kFooter = 104, kPad = 16, kMargin = 28, kLine = 27;

void appendUtf8(std::string & s, uint32_t c) {
	if (c < 0x80) s += (char)c;
	else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
	else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
	else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}
} // namespace

ofApp::ofApp(int argc, char ** argv) {
	opt.system = garmentSystemPrompt();
	opt.maxNewTokens = 3072;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
		if (a == "--prompt") cliPrompt = next();
		else if (a == "--out") outPath = next();
		else if (a == "--cpu") opt.useCuda = false;
		else if (a == "--cudnn") opt.cudaDllPath = next();
		else if (a == "--model") opt.modelDir = next();
		else if (a == "--max") opt.maxNewTokens = std::max(1, ofToInt(next()));
		else if (a == "--greedy") opt.temperature = 0;
	}
}

// Asks for a design and checks it. A reply that cannot be repaired (see extractDesignJson) is sent back to the model
// twice at most, with what was wrong; onRetry is called before each new attempt so the caller can clear its display.
// Returns the valid JSON, or "" if the model never produced one; `note` says what was repaired or what failed.
std::string ofApp::design(const std::string & prompt, const std::function<void(const std::string &)> & onToken,
	const std::function<void()> & onRetry, std::string & note) {
	std::string ask = garmentInstruction(prompt), json;
	for (int attempt = 0; attempt < 3; attempt++) {
		std::string answer;
		chat.chat(ask, [&](const std::string & t) { answer += t; onToken(t); });
		json = extractDesignJson(answer, &note);
		if (!json.empty() || chat.cancelled()) break;
		ask = "That was not a valid design JSON object (" + note + "). Reply again with only the corrected JSON object.";
		if (attempt < 2) onRetry();
	}
	return json;
}

void ofApp::loadFonts() {
	const std::vector<std::string> sans = { "C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
		"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "verdana.ttf" };
	const std::vector<std::string> bold = { "C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
		"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "verdana.ttf" };
	const std::vector<std::string> mono = { "C:/Windows/Fonts/CascadiaMono.ttf", "C:/Windows/Fonts/consola.ttf",
		"/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" };
	auto load = [](ofTrueTypeFont & f, const std::vector<std::string> & paths, int size) {
		for (auto & p : paths) {
			ofTrueTypeFontSettings s(p, size);
			s.antialiased = true;
			s.contours = false;
			s.addRanges(ofAlphabet::Latin);
			s.addRange(ofUnicode::Greek);
			s.addRange(ofUnicode::MathOperators);
			s.addRange(ofUnicode::Arrows);
			if (f.load(s)) return true;
		}
		return false;
	};
	fontsOk = load(fBody, sans, 18) && load(fBold, bold, 18) && load(fTitle, bold, 22) && load(fSmall, sans, 13) && load(fMono, mono, 13);
	if (!fontsOk) ofLogError("ofApp") << "no TrueType font found; text will not draw";
}

void ofApp::setup() {
	ofSetEscapeQuitsApp(false);
	ofSetVerticalSync(true);
	ofEnableAntiAliasing();
	ofSetCircleResolution(48);
	ofSetBackgroundColor(kBg);

	if (!cliPrompt.empty()) {
		// one question, no window use: load, answer, print, exit
		if (!chat.load(opt)) { ofExit(1); cliDone = true; return; }
		std::cerr << "[" << (chat.isGpu() ? "GPU (CUDA)" : "CPU") << "]" << std::endl;
		std::string note;
		std::string json = design(cliPrompt, [](const std::string &) {}, []() { std::cerr << "not valid, asking again" << std::endl; }, note);
		std::cerr << ofToString(chat.tokensPerSecond(), 1) << " tokens/s, " << chat.contextUsed() << " tokens of context" << std::endl;
		if (!json.empty() && !note.empty()) std::cerr << "repaired: " << note << std::endl;
		if (json.empty()) {
			std::cerr << "no valid design JSON: " << note << std::endl;
			cliDone = true;
			ofExit(2);
			return;
		}
		std::cout << json << std::endl;
		if (!outPath.empty()) std::ofstream(outPath) << json << std::endl;
		cliDone = true;
		ofExit(0);
		return;
	}

	loadFonts();
	loader = std::thread([this]() { state = chat.load(opt) ? 1 : 2; });
}

void ofApp::exit() {
	chat.cancel();
	if (worker.joinable()) worker.join();
	if (loader.joinable()) loader.join();
}

void ofApp::update() {
	if (worker.joinable() && !busy) worker.join();
}

void ofApp::send(const std::string & text) {
	if (!ready() || busy || text.empty()) return;
	if (worker.joinable()) worker.join();
	{
		std::lock_guard<std::mutex> lock(mtx);
		msgs.push_back({ true, text });
		msgs.push_back({ false, "" });
	}
	scroll = 0;
	busy = true;
	worker = std::thread([this, text]() {
		std::string note;
		std::string json = design(text, [this](const std::string & piece) {
			std::lock_guard<std::mutex> lock(mtx);
			msgs.back().text += piece;
		}, [this]() {
			std::lock_guard<std::mutex> lock(mtx);
			msgs.back().text.clear(); // the model is asked again; start the bubble over
		}, note);
		{
			// the answer is a valid design: show it as formatted JSON and keep a copy
			if (!json.empty()) {
				std::ofstream(ofToDataPath("designs/" + ofGetTimestampString("%Y%m%d_%H%M%S") + ".json", true)) << json << std::endl;
				std::lock_guard<std::mutex> lock(mtx);
				msgs.back().text = json;
			}
		}
		busy = false;
	});
}

void ofApp::stopAnswer() { if (busy) chat.cancel(); }

void ofApp::newChat() {
	if (busy || !ready()) return;
	std::lock_guard<std::mutex> lock(mtx);
	msgs.clear();
	chat.reset();
	scroll = 0;
}

void ofApp::keyPressed(ofKeyEventArgs & e) {
	if (e.key == OF_KEY_ESC) { stopAnswer(); return; }
	if (e.key == OF_KEY_F1) { newChat(); return; }
	if (e.key == OF_KEY_PAGE_UP) { scroll += viewH * 0.8f; return; }
	if (e.key == OF_KEY_PAGE_DOWN) { scroll -= viewH * 0.8f; return; }
	if (e.key == OF_KEY_RETURN) {
		std::string t = ofTrim(input);
		input.clear();
		send(t);
		return;
	}
	if (e.key == OF_KEY_BACKSPACE) {
		while (!input.empty() && (input.back() & 0xC0) == 0x80) input.pop_back();
		if (!input.empty()) input.pop_back();
		return;
	}
	if (e.codepoint >= 32 && e.codepoint != 127 && !(e.modifiers & (OF_KEY_CONTROL | OF_KEY_ALT))) appendUtf8(input, e.codepoint);
}

void ofApp::mouseScrolled(int, int, float, float sy) { scroll += sy * 70.f; }

void ofApp::mousePressed(int x, int y, int button) {
	if (button != 0) return;
	for (auto & c : chips)
		if (c.second.inside(x, y)) { send(c.first); return; }
}

// Wrap `text` to maxW pixels. For the assistant, <think>...</think> is cut out of the text and its lines flagged.
void ofApp::layout(const std::string & text, float maxW, bool assistant, Block & out) {
	out.user = !assistant;
	out.lines.clear();

	std::vector<std::pair<std::string, bool>> segs;
	if (assistant) {
		bool think = false;
		size_t pos = 0;
		while (pos <= text.size()) {
			size_t a = text.find(think ? "</think>" : "<think>", pos);
			segs.push_back({ text.substr(pos, a == std::string::npos ? std::string::npos : a - pos), think });
			if (a == std::string::npos) break;
			pos = a + (think ? 8 : 7);
			think = !think;
		}
	} else {
		segs.push_back({ text, false });
	}

	auto wrap = [&](const std::string & para, bool think) {
		std::vector<std::string> words = ofSplitString(para, " ", false, false);
		std::string cur;
		for (auto & w : words) {
			std::string cand = cur.empty() ? w : cur + " " + w;
			if (cur.empty() || fBody.stringWidth(cand) <= maxW) { cur = cand; continue; }
			out.lines.push_back({ cur, think });
			cur = w;
		}
		// a single word wider than the box: break it by characters
		while (fBody.stringWidth(cur) > maxW && cur.size() > 1) {
			size_t n = cur.size();
			while (n > 1 && fBody.stringWidth(cur.substr(0, n)) > maxW) {
				n--;
				while (n > 1 && (cur[n] & 0xC0) == 0x80) n--;
			}
			out.lines.push_back({ cur.substr(0, n), think });
			cur = cur.substr(n);
		}
		out.lines.push_back({ cur, think });
	};

	for (auto & seg : segs) {
		std::string t = seg.first;
		t.erase(std::remove(t.begin(), t.end(), '\r'), t.end());
		size_t b = t.find_first_not_of("\n "), e = t.find_last_not_of("\n ");
		if (b == std::string::npos) continue;
		t = t.substr(b, e - b + 1);
		for (auto & para : ofSplitString(t, "\n", false, false)) wrap(para, seg.second);
		if (&seg != &segs.back()) out.lines.push_back({ "", seg.second }); // gap after a thinking block
	}
	if (assistant && !out.lines.empty() && out.lines.back().s.empty()) out.lines.pop_back();

	out.w = 0;
	for (auto & l : out.lines) out.w = std::max(out.w, l.s.empty() ? 0.f : fBody.stringWidth(l.s));
	out.h = out.lines.size() * kLine;
}

void ofApp::drawChip(const std::string & s, float right, float y, const ofColor & dot, float & usedW) {
	float tw = fSmall.stringWidth(s);
	float w = tw + 34, h = 28;
	float x = right - w;
	ofSetColor(kPanel2);
	ofDrawRectRounded(x, y, w, h, 14);
	ofSetColor(dot);
	ofDrawCircle(x + 14, y + h / 2, 4);
	ofSetColor(kText);
	fSmall.drawString(s, x + 26, y + h / 2 + 5);
	usedW = w + 8;
}

void ofApp::drawBubble(const Block & b, float y, float areaW, bool streaming) {
	const float w = std::max(b.w + 2 * kPad, 56.f), h = b.h + 2 * kPad - 4;
	float x = b.user ? areaW - kMargin - w : kMargin + 40;
	if (b.user) {
		ofSetColor(kUser);
		ofDrawRectRounded(x, y, 0, w, h, 18, 18, 4, 18);
	} else {
		ofSetColor(kPanel);
		ofDrawRectRounded(x, y, 0, w, h, 18, 4, 18, 18);
		ofSetColor(kBorder);
		ofNoFill();
		ofDrawRectRounded(x, y, 0, w, h, 18, 4, 18, 18);
		ofFill();
		// avatar
		ofSetColor(kAccent);
		ofDrawCircle(kMargin + 15, y + 17, 15);
		ofSetColor(255);
		fBold.drawString("P", kMargin + 15 - fBold.stringWidth("P") / 2, y + 23);
	}
	float ty = y + kPad + 15;
	bool inThink = false;
	float thinkTop = 0;
	for (size_t i = 0; i < b.lines.size(); i++) {
		auto & l = b.lines[i];
		if (l.think) {
			if (!inThink) { inThink = true; thinkTop = ty - 19; }
			ofSetColor(kMuted);
		} else {
			if (inThink) { inThink = false; ofSetColor(kBorder); ofDrawRectangle(x + 8, thinkTop, 2.5f, ty - 19 - thinkTop - 4); }
			ofSetColor(b.user ? ofColor(255) : kText);
		}
		fBody.drawString(l.s, x + kPad + (l.think ? 6 : 0), ty);
		ty += kLine;
	}
	if (inThink) { ofSetColor(kBorder); ofDrawRectangle(x + 8, thinkTop, 2.5f, ty - 19 - thinkTop - 4); }

	if (streaming) {
		// blinking caret after the last line, or three dots while nothing has arrived yet
		if (b.lines.empty()) {
			for (int i = 0; i < 3; i++) {
				float a = 0.35f + 0.65f * (0.5f + 0.5f * std::sin(ofGetElapsedTimef() * 6.f - i * 0.9f));
				ofSetColor(kMuted, 255 * a);
				ofDrawCircle(x + kPad + 6 + i * 14, y + h / 2, 3.5f);
			}
		} else if (fmod(ofGetElapsedTimef(), 1.0f) < 0.6f) {
			float cx = x + kPad + fBody.stringWidth(b.lines.back().s) + (b.lines.back().think ? 8 : 3);
			ofSetColor(kAccent);
			ofDrawRectangle(cx, ty - kLine - 14, 2, 20);
		}
	}
}

void ofApp::draw() {
	if (cliDone || !fontsOk) return;
	const float W = ofGetWidth(), H = ofGetHeight();
	const float t = ofGetElapsedTimef();

	// header
	ofSetColor(kPanel);
	ofDrawRectangle(0, 0, W, kHeader);
	ofSetColor(kBorder);
	ofDrawRectangle(0, kHeader - 1, W, 1);
	ofSetColor(kAccent);
	ofDrawCircle(kMargin + 14, kHeader / 2, 14);
	ofSetColor(255);
	fBold.drawString("P", kMargin + 14 - fBold.stringWidth("P") / 2, kHeader / 2 + 6);
	ofSetColor(kText);
	fTitle.drawString("Garment design", kMargin + 40, kHeader / 2 + 2);
	ofSetColor(kMuted);
	fSmall.drawString("ONNX Runtime  \xC2\xB7  ofxOnnx", kMargin + 41, kHeader / 2 + 20);

	if (ready()) {
		float used = 0, right = W - kMargin;
		drawChip(chat.isGpu() ? "GPU \xC2\xB7 CUDA" : "CPU", right, 18, chat.isGpu() ? kGreen : kAmber, used);
		right -= used;
		if (chat.tokensPerSecond() > 0) {
			drawChip(ofToString(chat.tokensPerSecond(), 1) + " tok/s", right, 18, kAccent, used);
			right -= used;
		}
		// context meter
		float frac = ofClamp(chat.contextUsed() / (float)chat.contextMax(), 0, 1);
		ofSetColor(kPanel2);
		ofDrawRectRounded(right - 110, 29, 100, 6, 3);
		ofSetColor(frac > 0.85f ? kAmber : kAccent);
		ofDrawRectRounded(right - 110, 29, std::max(6.f, 100 * frac), 6, 3);
		ofSetColor(kMuted);
		fSmall.drawString("context", right - 110, 52);
	}

	if (state == 0 || state == 2) {
		// loading / failed
		ofPoint c(W / 2, H / 2 - 20);
		if (state == 0) {
			// a big spinner: a ring of dots with a fading tail, sized to the window
			const int n = 16;
			const float R = std::max(70.f, std::min(W, H) * 0.16f), dot = R * 0.15f;
			for (int i = 0; i < n; i++) {
				float a = fmod(t * 1.2f - i / (float)n, 1.f);
				if (a < 0) a += 1;
				ofSetColor(kAccent, 40 + 215 * (1 - a));
				float ang = TWO_PI * i / n - HALF_PI;
				ofDrawCircle(c.x + std::cos(ang) * R, c.y + std::sin(ang) * R, dot * (0.45f + 0.55f * (1 - a)));
			}
			ofSetColor(kText);
			std::string s = opt.useCuda ? "Loading the model onto the GPU" : "Loading the model";
			fTitle.drawString(s, c.x - fTitle.stringWidth(s) / 2, c.y + R + 56);
			ofSetColor(kMuted);
			s = "a few seconds the first time";
			fBody.drawString(s, c.x - fBody.stringWidth(s) / 2, c.y + R + 86);
		} else {
			ofSetColor(kAmber);
			std::string s = "The model did not load. See the console for the reason.";
			fBody.drawString(s, c.x - fBody.stringWidth(s) / 2, c.y);
		}
		return;
	}

	// transcript
	const float top = kHeader, bottom = H - kFooter;
	viewH = bottom - top;
	const float maxBubble = std::min(W - 2 * kMargin - 40, 780.f) - 2 * kPad;
	std::vector<Block> blocks;
	bool streaming;
	{
		std::lock_guard<std::mutex> lock(mtx);
		blocks.resize(msgs.size());
		for (size_t i = 0; i < msgs.size(); i++) layout(msgs[i].text, msgs[i].user ? std::min(maxBubble, W * 0.7f) : maxBubble, !msgs[i].user, blocks[i]);
		streaming = busy && !msgs.empty();
	}

	chips.clear();
	ofRectangle clip(0, top, W, viewH);
	ofSetColor(255);
	ofRectangle prevClip;
	glEnable(GL_SCISSOR_TEST);
	glScissor(0, (int)kFooter, (int)W, (int)viewH);

	if (blocks.empty()) {
		ofSetColor(kText);
		std::string s = "Ask me something";
		fTitle.drawString(s, W / 2 - fTitle.stringWidth(s) / 2, top + viewH / 2 - 50);
		ofSetColor(kMuted);
		s = "Describe a garment and get its GarmentCode design as JSON.";
		fBody.drawString(s, W / 2 - fBody.stringWidth(s) / 2, top + viewH / 2 - 18);
		const std::vector<std::string> ex = { "a knee-length sleeveless summer dress with a V neckline", "a long-sleeve shirt", "a pencil skirt with a waistband" };
		float tw = 0;
		for (auto & e : ex) tw += fSmall.stringWidth(e) + 36 + 10;
		float x = W / 2 - (tw - 10) / 2, y = top + viewH / 2 + 12;
		for (auto & e : ex) {
			float w = fSmall.stringWidth(e) + 36;
			ofRectangle r(x, y, w, 36);
			bool hot = r.inside(ofGetMouseX(), ofGetMouseY());
			ofSetColor(hot ? kBorder : kPanel2);
			ofDrawRectRounded(r, 18);
			ofSetColor(hot ? kAccent : kBorder);
			ofNoFill();
			ofDrawRectRounded(r, 18);
			ofFill();
			ofSetColor(kText);
			fSmall.drawString(e, x + 18, y + 23);
			chips.push_back({ e, r });
			x += w + 10;
		}
	} else {
		contentH = kPad;
		for (auto & b : blocks) contentH += b.h + 2 * kPad - 4 + 16;
		scroll = ofClamp(scroll, 0, std::max(0.f, contentH - viewH));
		float y = contentH < viewH ? top + kPad : bottom - contentH + scroll + kPad;
		for (size_t i = 0; i < blocks.size(); i++) {
			float bh = blocks[i].h + 2 * kPad - 4;
			if (y + bh > top - 10 && y < bottom + 10) drawBubble(blocks[i], y, W, streaming && i + 1 == blocks.size() && !blocks[i].user);
			y += bh + 16;
		}
	}
	glDisable(GL_SCISSOR_TEST);

	// fade the edges of the transcript
	ofMesh fade;
	fade.setMode(OF_PRIMITIVE_TRIANGLE_STRIP);
	for (int k = 0; k < 2; k++) {
		float y0 = k == 0 ? top : bottom - 18, y1 = k == 0 ? top + 18 : bottom;
		ofColor a = k == 0 ? ofColor(kBg, 255) : ofColor(kBg, 0), b = k == 0 ? ofColor(kBg, 0) : ofColor(kBg, 255);
		fade.clear();
		fade.addVertex({ 0, y0, 0 }); fade.addColor(a);
		fade.addVertex({ W, y0, 0 }); fade.addColor(a);
		fade.addVertex({ 0, y1, 0 }); fade.addColor(b);
		fade.addVertex({ W, y1, 0 }); fade.addColor(b);
		fade.draw();
	}

	// input
	const float bx = kMargin, bw = W - 2 * kMargin, by = H - kFooter + 14, bh = 52;
	ofSetColor(kPanel);
	ofDrawRectRounded(bx, by, bw, bh, 26);
	ofSetColor(busy ? kBorder : kAccent);
	ofNoFill();
	ofDrawRectRounded(bx, by, bw, bh, 26);
	ofFill();
	const float inner = bw - 80;
	float tx = bx + 24;
	if (input.empty()) {
		ofSetColor(kMuted);
		fBody.drawString(busy ? "Designing. Esc to stop." : "Describe a garment\xE2\x80\xA6", tx, by + 32);
	} else {
		float w = fBody.stringWidth(input);
		float off = std::max(0.f, w - inner);
		glEnable(GL_SCISSOR_TEST);
		glScissor((int)tx, (int)(H - by - bh), (int)inner + 6, (int)bh);
		ofSetColor(kText);
		fBody.drawString(input, tx - off, by + 32);
		glDisable(GL_SCISSOR_TEST);
		tx += w - off;
	}
	if (!busy && fmod(t, 1.0f) < 0.6f) {
		ofSetColor(kAccent);
		ofDrawRectangle(tx + (input.empty() ? 0 : 2), by + 14, 2, 24);
	}
	// send button
	const bool can = !busy && !ofTrim(input).empty();
	ofSetColor(can ? kAccent : kPanel2);
	ofDrawCircle(bx + bw - 28, by + bh / 2, 17);
	ofSetColor(can ? ofColor(255) : kMuted);
	ofDrawTriangle(bx + bw - 33, by + bh / 2 - 8, bx + bw - 33, by + bh / 2 + 8, bx + bw - 20, by + bh / 2);

	ofSetColor(kMuted);
	fSmall.drawString("Enter send   \xC2\xB7   Esc stop   \xC2\xB7   F1 new chat   \xC2\xB7   wheel scroll", bx + 8, H - 12);
	if (chat.contextUsed() > 0) {
		std::string s = ofToString(chat.contextUsed()) + " / " + ofToString(chat.contextMax()) + " tokens";
		fSmall.drawString(s, bx + bw - fSmall.stringWidth(s) - 8, H - 12);
	}
}
