#include "ofMain.h"
#include "ofApp.h"

int main(int argc, char** argv) {
    // --check: run the comparison, print PASS/FAIL, and exit with 0 or 1 (for scripts and CI).
    bool check = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--check") check = true;

    ofGLWindowSettings settings;
    settings.setSize(900, 520);
    settings.setGLVersion(3, 2);
    ofCreateWindow(settings);
    return ofRunApp(new ofApp(check));
}
