#include "ofApp.h"
#include "ofMain.h"

int main(int argc, char ** argv) {
	ofGLWindowSettings settings;
	settings.setSize(1000, 760);
	settings.setGLVersion(3, 2);
	settings.title = "Phi-4 on ofxOnnx";
	ofCreateWindow(settings);
	return ofRunApp(new ofApp(argc, argv));
}
