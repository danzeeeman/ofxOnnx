#include "ofMain.h"
#include "ofApp.h"

int main(int argc, char** argv) {
    // --check [seconds]: no window needed to judge it: walk forward for that many simulated seconds, print how it
    // went, and exit 0 (it walked and stayed up) or 1.
    bool check = false;
    double seconds = 10.0;
    int ducks = 3;              // --ducks N: the robot you drive and N - 1 followers (--solo is 1)
    bool solo = false;          // --solo: one robot, no followers
    bool run = false;           // --run: the Run policy
    bool threaded = false;      // --threads: with --check, run the two threads in real time instead of stepping in a loop
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--check") {
            check = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') seconds = std::atof(argv[++i]);
        }
        if (std::string(argv[i]) == "--threads") threaded = true;
        if (std::string(argv[i]) == "--run") run = true;
        if (std::string(argv[i]) == "--solo") solo = true;
        if (std::string(argv[i]) == "--ducks" && i + 1 < argc) ducks = std::max(1, std::atoi(argv[++i]));
    }

    ofGLWindowSettings settings;
    settings.setSize(1280, 800);
    settings.setGLVersion(3, 2);
    ofCreateWindow(settings);
    return ofRunApp(new ofApp(check, seconds, threaded, run, solo ? 1 : ducks));
}
