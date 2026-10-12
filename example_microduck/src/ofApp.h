#pragma once
#include "ofMain.h"
#include "Microduck.h"
#include "CameraPath.h"       // ofxRobot/src/core
#include "FrameRecorder.h"

// A trained walking policy making a biped walk: the microduck, in MuJoCo, driven by ofxOnnx, described and checked
// with ofxRobot. See Microduck.h for who does what.
//
//   1 / 2             the Walk policy / the Run policy
// Walk:
//   up / down         walk forward / back (velocity command, m/s)
//   left / right      turn
//   , and .           step sideways
//   space             stop (zero command)
// Run (hold the keys):
//   up / down         move the target away from the robot / toward it
//   left / right      move the target to the robot's left / right
//   space             put the target 2.5 m ahead again
//   x                 shove it
//   r                 reset
//   p                 pause
//   k                 show ofxRobot's kinematics: each limb's joints, and the foot where its forward kinematics puts it
//   g                 show the collision geometry
//   f                 clear a safety-gate fault
//   drag, wheel       orbit / zoom the camera (it follows the group)
//
// Two more microducks (--ducks N for N in all) run the Follow policy: the first keeps 0.6 m behind the one you drive, the
// second behind the first, and so on. They are separate simulations in one coordinate frame, so they do not collide with each other.
//
// Command line: --check [seconds] walks forward for that long without drawing anything and exits 0 or 1;
// --check --threads does it in real time on the physics and inference threads;
// --ducks N is N robots in all (3 by default), --solo one robot with no followers; --run starts with the Run policy (with --check, it has to run to a target 3 m ahead and stop there).
class ofApp : public ofBaseApp {
public:
    ofApp(bool check = false, double checkSeconds = 10.0, bool threaded = false, bool run = false, int ducks = 3, bool simplify = false)
        : _check(check), _checkSeconds(checkSeconds), _threaded(threaded), _startRun(run), _chain(ducks > 1), _followerCount(ducks - 1), _simplify(simplify) {}

    void setup() override;
    void update() override;
    void exit() override;
    void draw() override;
    void keyPressed(int key) override;

    // --render [dir] [--fps n] [--size WxH] [--seconds s]: write a 30 second camera flight round the group as PNG frames, frame by
    // frame, and exit (ofxRobot's FrameRecorder.h; 1920x1080, 50 fps by default: one frame per policy period, to C:/src/renders/
    // example_microduck unless a folder is given). The walker walks a circle by itself and the followers trail it; nothing runs in
    // real time (no threads), so the video is the same however long a frame takes to draw.
    FrameRecorder::Options render;

private:
    void drawScene();
    void runCheck();
    void runChainCheck();
    void drawRobots();                  // every robot, one instanced draw per mesh
    void drawKinematics(Microduck& duck);   // one robot's limbs: its joints, and the foot by forward kinematics and by the simulation
    glm::mat4 toMat4(const double* pos, const double* mat) const;

    bool _check = false;
    double _checkSeconds = 10.0;
    bool _threaded = false;
    bool _startRun = false;
    bool _chain = true;                  // more robots, each following the one ahead of it
    bool _simplify = false;              // --simplify: reduced display meshes
    int _followerCount = 2;              // so 3 robots in all by default

    Microduck _duck;                                    // the one you drive
    std::vector<std::unique_ptr<Microduck>> _followers;  // each follows the robot ahead of it: the first follows _duck
    std::vector<ofVboMesh> _meshes;      // every mesh in the model, in the model's own frame
    ofEasyCam _cam;
    ofCamera _flightCam;                 // the camera of a recording: placed by the flight, not the mouse
    FrameRecorder _rec;
    CameraPath _flight;
    double _flightT = 0.0, _simCarry = 0.0;
    ofShader _shader;                    // lit, with one transform and colour per instance (per robot)
    ofLight _light;
    ofPolyline _trail;
    glm::vec3 _focus{0.f};

    float _vx = 0.2f, _vy = 0.f, _wz = 0.f;   // it starts out walking straight
    bool _showKinematics = true, _showCollision = false;
    double _realtime = 1.0;
    double _lastWall = 0.0, _lastSim = 0.0;
};
