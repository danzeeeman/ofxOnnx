#include "ofMain.h"
#include "ofApp.h"

int main(int argc, char** argv) {
    // --check [seconds]: no window needed to judge it: walk forward for that many simulated seconds, print how it went, and
    // exit 0 (it walked and stayed up) or 1. --threads: the same in real time on the physics and inference threads.
    // --parity: the two networks through ofxOnnx against PyTorch's answers; exit 0 or 1.
    // --record / --render <dir> [--fps n] [--size WxH] [--seconds s]: PNG frames of a scripted walk; --lock: a fixed camera.
    // --hold: do not turn streaming on by itself. --publish [port]: each leg's RobotState as JSON over TCP / UDP (StatePublisher).
    ofApp::Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--check") {
            opt.check = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') opt.checkSeconds = std::atof(argv[++i]);
        }
        if (a == "--threads") opt.threaded = true;
        if (a == "--parity") opt.parity = true;
        if (a == "--lock") opt.lock = true;
        if (a == "--hold") opt.hold = true;
        if (a == "--kin") opt.kin = true;
        if (a == "--collision") opt.collision = true;
        if (a == "--walk" && i + 1 < argc) opt.walkSpeed = std::atof(argv[++i]);
        if (a == "--shot" && i + 1 < argc) opt.shot = argv[++i];
        if (a == "--publish") {
            opt.publishPort = 7001;
            if (i + 1 < argc && argv[i + 1][0] != '-') opt.publishPort = std::atoi(argv[++i]);
        }
    }

    ofGLFWWindowSettings settings;
    settings.setSize(1280, 800);
    settings.setGLVersion(3, 2);
    settings.numSamples = 4;
    ofCreateWindow(settings);
    auto* app = new ofApp(opt);
    app->render = FrameRecorder::Options::parse(argc, argv);   // --record / --render [dir] [--fps n] [--size WxH] [--seconds s]
    return ofRunApp(app);
}
