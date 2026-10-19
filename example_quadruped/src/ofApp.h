#pragma once
#include "ofMain.h"
#include "QuadrupedClient.h"
#include "QuadrupedBrain.h"
#include "DomeLight.h"        // ofxRobot/src/core
#include "BrandColour.h"
#include "FrameRecorder.h"
#include "LabelBatch.h"
#include "RobotMesh.h"
#include "StatePublisher.h"
#include <deque>

// A trained walking policy making a quadruped walk, built the way ofxRobot builds every robot: the ANYmal-D as a streaming
// client (QuadrupedClient: MuJoCo, and the ANYdrive actuator network through ofxOnnx, behind the same calls as an arm's client),
// the walking policy as the app's side of the loop (QuadrupedBrain: ofxOnnx, writing joint targets into RobotState), the legs as
// JSON descriptors with forward kinematics (RobotKinematics, ArmIk), every target through MotionSafetyGate, and the robot drawn
// from its meshes (RobotMesh) under a DomeLight. See the README for who does what.
//
//   up / down         walk forward / back (velocity command, m/s)
//   left / right      turn
//   , and .           step sideways
//   space             stop (zero command)
//   s                 streaming on / off (off: the legs hold the stand pose)
//   x                 shove it
//   r                 reset (stand it up again)
//   p                 pause
//   k                 show ofxRobot's kinematics: each leg's joints, and the foot where its forward kinematics puts it
//   g                 show the collision geometry
//   f                 clear a safety-gate fault (streaming stays off until you press s)
//   drag, wheel       orbit / zoom the camera (it follows the robot)
//
// Command line:
//   --check [seconds]   walk forward at 0.5 m/s for that long without drawing anything and exit 0 or 1, after checking that the
//                       legs hold the stand pose with streaming off and that a bad target stops streaming; --threads: the
//                       walk in real time on the controller's worker thread
//   --parity            run the stored reference observations through the policy and the stored actuator steps through the
//                       actuator network (both via ofxOnnx) and compare them with PyTorch's answers; exit 0 or 1
//   --record <dir> [--fps n] [--size WxH] [--seconds s] [--lock]
//                       write PNG frames of a scripted walk (stand, forward, turn round, back, sideways, back up, stand), frame
//                       by frame (ofxRobot's FrameRecorder), and exit; --render <dir> is the same. The world is stepped by the
//                       frame time, not the clock, so the video does not depend on how fast the machine draws. --lock: a
//                       fixed camera beside the path instead of one that follows the robot.
//   --walk <m/s>        the recording's walking speed (default 0.5; 1.0 is the fastest it was trained for)
//   --shot <file> [--kin] [--collision]   save a screenshot of the window (HUD and overlays included) a few seconds in, and exit
//   --hold              do not turn streaming on by itself (the legs stand until you press s)
//   --publish [port]    publish each leg's RobotState with StatePublisher (JSON over TCP and UDP): leg k on TCP port + 2k, UDP
//                       port + 2k + 1 (default 7001: LF 7001/7002, LH 7003/7004, RF 7005/7006, RH 7007/7008)
class ofApp : public ofBaseApp {
public:
    struct Options {
        bool check = false;
        double checkSeconds = 10.0;
        bool threaded = false;
        bool parity = false;
        bool lock = false;
        bool hold = false;
        int publishPort = 0;                    // 0: not publishing
        double walkSpeed = 0.5;                 // --walk <m/s>: the recording's forward speed (the other speeds scale with it)
        std::string shot;                       // --shot <file>: save the window once the robot is walking, and exit
        bool kin = false, collision = false;    // --kin / --collision: start with the overlays on
    };
    explicit ofApp(const Options& o) : _opt(o) {}

    void setup() override;
    void update() override;
    void exit() override;
    void draw() override;
    void keyPressed(int key) override;
    void mouseDragged(int x, int y, int button) override;
    void mouseScrolled(int x, int y, float scrollX, float scrollY) override;

    FrameRecorder::Options render;

private:
    void drawScene();
    void drawFloor(const QuadrupedClient::Sense& s);
    void drawRobot();
    void drawCollision();
    void drawKinematics(const QuadrupedClient::Sense& s);
    void drawHud();
    bool connectClient(bool realtime);
    void runParity();
    void runCheck();
    void exchange();                             // brain -> RobotState -> client -> RobotState, once; the publishers
    void trackFeet(const QuadrupedClient::Sense& s);
    void scriptCommand(double t);                // the recording's walk
    void placeCamera(const glm::vec3& target, float azDeg, float elDeg, float dist);

    Options _opt;
    std::array<RobotState, QuadrupedClient::kLegs> _legState;   // one per leg: a JointVec holds up to 7 joints, the robot has 12
    std::array<RobotState*, QuadrupedClient::kLegs> _legs;
    QuadrupedClient _client;
    QuadrupedBrain _brain;
    std::array<StatePublisher, QuadrupedClient::kLegs> _pub;
    bool _publishing = false;

    // Drawing.
    struct Part { int body = -1; ofVboMesh mesh; ofMaterial material; };
    std::vector<Part> _parts;
    DomeLight _dome;
    ofTrueTypeFont _font;
    LabelBatch _labels;
    ofCamera _cam;
    FrameRecorder _rec;
    std::vector<glm::mat4> _bodyPose, _geomPose;
    std::array<std::vector<int>, QuadrupedClient::kLegs> _footGeoms;   // each foot's collision spheres, to see when it is on the floor
    struct Step { glm::vec2 pos; double t; int leg; };
    std::deque<Step> _steps;                     // where the feet came down, fading
    std::array<bool, QuadrupedClient::kLegs> _contact{};
    ofVboMesh _grid;

    // The camera follows the robot, damped, three quarters from the front; the mouse changes the offsets.
    glm::vec3 _focus{0.f};
    double _camHeading = 0.0;
    float _azOffset = -42.f, _el = 17.f, _dist = 3.0f;
    bool _focusInit = false;

    float _vx = 0.4f, _vy = 0.f, _wz = 0.f;       // the keys' command: it starts out walking straight
    bool _showKinematics = false, _showCollision = false;
    bool _streamed = false;                      // streaming has been turned on (by itself, or by the operator)
    double _streamAt = 1.0;                      // sim time at which streaming turns itself on (the robot has settled)
    double _realtime = 1.0, _lastSim = 0.0;
    double _carry = 0.0;                         // a recording: physics steps owed
    int _lastResets = 0;
};
