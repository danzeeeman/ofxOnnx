#include "ofMain.h"
#include "ofApp.h"

int main(int argc, char** argv) {
    // --check [seconds]: no window needed to judge it: walk forward for that many simulated seconds, print how it
    // went, and exit 0 (it walked and stayed up) or 1.
    bool check = false;
    double seconds = 10.0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--check") {
            check = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') seconds = std::atof(argv[++i]);
        }
    }

    ofGLWindowSettings settings;
    settings.setSize(1280, 800);
    settings.setGLVersion(3, 2);
    ofCreateWindow(settings);
    return ofRunApp(new ofApp(check, seconds));
}
