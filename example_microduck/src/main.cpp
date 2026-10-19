#include "ofMain.h"
#include "ofApp.h"
#include "SwarmApp.h"

int main(int argc, char** argv) {
    // --check [seconds]: no window needed to judge it: walk forward for that many simulated seconds, print how it
    // went, and exit 0 (it walked and stayed up) or 1.
    bool check = false;
    double seconds = 10.0;
    int ducks = 3;              // --ducks N: the robot you drive and N - 1 followers (--solo is 1)
    bool simplify = false;      // --simplify: draw reduced meshes (a lighter load for the GPU)
    bool swarm = false;         // --swarm: twelve ducks in one world, avoiding each other (SwarmApp)
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
        if (std::string(argv[i]) == "--swarm") swarm = true;
        if (std::string(argv[i]) == "--simplify") simplify = true;
        if (std::string(argv[i]) == "--ducks" && i + 1 < argc) ducks = std::max(1, std::atoi(argv[++i]));
    }

    ofGLWindowSettings settings;
    settings.setSize(1280, 800);
    settings.setGLVersion(3, 2);
    ofCreateWindow(settings);
    if (swarm) {
        auto* app = new SwarmApp(check, check && seconds == 10.0 ? 30.0 : seconds, simplify);
        app->render = FrameRecorder::Options::parse(argc, argv);   // --render [dir] [--fps n] [--size WxH] [--seconds s]
        bool fpsGiven = false;
        for (int i = 1; i < argc; ++i) fpsGiven = fpsGiven || std::string(argv[i]) == "--fps";
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--track") app->track = true;
        if (!fpsGiven) app->render.fps = 50;                       // the swarm's own rate: one frame per policy period
        return ofRunApp(app);
    }
    auto* app = new ofApp(check, seconds, threaded, run, solo ? 1 : ducks, simplify);
    app->render = FrameRecorder::Options::parse(argc, argv);   // --render [dir] [--fps n] [--size WxH] [--seconds s]
    bool fpsGiven = false;
    for (int i = 1; i < argc; ++i) fpsGiven = fpsGiven || std::string(argv[i]) == "--fps";
    if (!fpsGiven) app->render.fps = 50;                       // the policy's own rate: one frame per period
    return ofRunApp(app);
}
