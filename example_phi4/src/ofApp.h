#pragma once
#include "Phi4Chat.h"
#include "ofMain.h"
#include <atomic>
#include <mutex>
#include <thread>

// Describe a garment and Phi-4 (running on the GPU through ofxOnnx) answers with its GarmentCode design as JSON,
// the same shape as garmentgen-demo's design.json. Designs are saved in bin/data/designs.
//
// Type a question and press Enter; the answer streams in token by token. The model "thinks" first, between
// <think> tags, and that part is drawn dimmed with a rule beside it.
//   Enter send   Esc stop the answer   F1 new chat   mouse wheel / PgUp / PgDn scroll
//
// Command line (no window interaction, prints the answer and the speed):
//   --prompt "text"   print the design JSON for one description and exit   --out file  also write it there      --cpu   run on the CPU
//   --model dir       a model folder under bin/data     --max n  most new tokens
//   --greedy          no sampling                  --cudnn dir  a folder holding cuDNN 9 for CUDA 13
class ofApp : public ofBaseApp {
public:
	ofApp(int argc, char ** argv);

	void setup() override;
	void update() override;
	void draw() override;
	void exit() override;
	void keyPressed(ofKeyEventArgs & e) override;
	void mousePressed(int x, int y, int button) override;
	void mouseScrolled(int x, int y, float sx, float sy) override;

private:
	struct Msg { bool user; std::string text; };
	struct Line { std::string s; bool think; };
	struct Block { bool user = false; std::vector<Line> lines; float w = 0, h = 0; };

	std::string design(const std::string & prompt, const std::function<void(const std::string &)> & onToken,
		const std::function<void()> & onRetry, std::string & note);
	void send(const std::string & text);
	void stopAnswer();
	void newChat();
	void layout(const std::string & text, float maxW, bool assistant, Block & out);
	void drawBubble(const Block & b, float y, float areaW, bool streaming);
	void drawChip(const std::string & s, float right, float y, const ofColor & dot, float & usedW);
	void loadFonts();
	bool ready() const { return state == 1; }

	Phi4Chat chat;
	Phi4Chat::Options opt;
	std::string cliPrompt, outPath;
	std::thread loader, worker;
	std::atomic<int> state { 0 }; // 0 loading, 1 ready, 2 failed
	std::atomic<bool> busy { false };
	std::mutex mtx;
	std::vector<Msg> msgs;
	std::string input;
	float scroll = 0, contentH = 0, viewH = 0;
	bool cliDone = false;

	ofTrueTypeFont fBody, fBold, fSmall, fTitle, fMono;
	bool fontsOk = false;
	std::vector<std::pair<std::string, ofRectangle>> chips; // starter prompts and their hit boxes
};
