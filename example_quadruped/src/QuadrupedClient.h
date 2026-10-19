#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"
#include "ArmIk.h"                 // ofxRobot: forward kinematics of a leg
#include "MotionSafetyGate.h"      // ofxRobot: sanity check on the joint targets that get sent
#include "RobotKinematics.h"       // ofxRobot: the legs, loaded from JSON descriptors
#include "RobotState.h"            // ofxRobot: the state the app and a controller exchange
#include <mujoco/mujoco.h>         // ofxRobot/libs/mujoco
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

// The ANYmal-D as a controller, in ofxRobot's streaming-client shape (see ofxRobot/AGENTS.md, "The streaming-client contract"):
// the same calls as EgmManager, LbrClient, UrClient, XArmClient and StandardBotsClient, so an app that drives an arm through one
// of those can drive this robot the same way, and could swap this class for a real controller's client.
//
// The "controller" here is MuJoCo (the robot's body and the floor) with the ANYdrive actuator network (an LSTM, run through
// ofxOnnx at every 5 ms physics step) as its motor drive. It runs in a worker thread, in real time. The brain that decides
// where the legs go, the walking policy, is NOT in here: it is the app's side (QuadrupedBrain), which writes joint targets
// into RobotState, exactly as an app writes targets for an arm.
//
// A JointVec holds up to 7 joints and this robot has 12, so it is four RobotStates, one per leg (LF, LH, RF, RH), each with the
// leg's three joints (HAA, HFE, KFE) in degrees, as in the leg descriptors.
//
//   setup(legs)         the four RobotStates the app owns
//   connect(config)     starts the worker and returns at once (it loads the model, then reports connected)
//   update()            once per frame, main thread: targetJoints / hasTarget in; currentJoints, commandedJoints, the connected
//                       flag, the time and the counters out, under a short lock. RobotState is touched only here.
//   consumeJustConnected()   true once per (re)connection: seed targetJoints from currentJoints
//   close()
//
// Enforced in the worker, whatever the app asks:
//   - streaming is OFF by default (setStreaming): until it is on, the legs hold the stand pose
//   - joint limits (setJointLimits) and a per-cycle speed cap (Config::maxJointDegPerSec) on every command
//   - MotionSafetyGate::observe on every leg's command; a fault (NaN, an impossible jump) stops streaming at once, holds the
//     stand pose, and stays until the operator clears it (clearFault) and turns streaming on again: nothing is cleared by itself
//
// The robot's body is not a joint, so besides the legs the controller reports what a real robot's IMU and state estimator would:
// sense(), a snapshot the brain reads.
//
// With Config::realtime false there is no worker: the caller steps the physics, one 5 ms step per stepOffline(), which is how a
// recording and the headless checks run (the same code, the same results whatever the machine's speed).
class QuadrupedClient {
public:
    static constexpr int kLegs = 4, kLegJoints = 3, kJoints = 12;

    struct Config {
        std::string dir = "anymal_d";           // under bin/data: anymal_d.xml, anymal_d.json, the leg descriptors, actuator.onnx
        bool realtime = true;                   // a worker thread, in real time; false: the caller steps it (stepOffline)
        double maxJointDegPerSec = 15000.0;     // the cap on how fast a command may move. The policy moves its targets by tens of
                                                // degrees per 20 ms policy period by design, so the cap only catches the absurd.
        double holdDegPerSec = 240.0;           // how fast the legs return to the stand pose when streaming stops
        double spawnX = 0.0, spawnY = 0.0, spawnYaw = 0.0;
        bool autoReset = true;                  // a robot that falls over is stood up again at the spawn point (and counted)
    };

    // What the controller reports about the body, in the policy's frames.
    struct Sense {
        bool valid = false;
        double simTime = 0.0;                   // s
        glm::dvec3 pos{0.0};                    // trunk, world
        glm::dmat3 rot{1.0};                    // body to world
        glm::dvec3 linVelBody{0.0};             // m/s, in the body frame
        glm::dvec3 angVelBody{0.0};             // rad/s, in the body frame
        glm::dvec3 gravityBody{0.0, 0.0, -1.0}; // unit vector, in the body frame
        std::array<float, kJoints> q{}, qd{};   // joint positions (rad) and velocities (rad/s), in the policy's order
    };

    // One leg, as ofxRobot sees it.
    struct Leg {
        std::string name;                       // LF, LH, RF, RH
        RobotKinematics kin;                    // from <name>.joints.json
        ArmIk ik;                               // forward kinematics from the same chain
        glm::dmat4 tool{1.0};                   // the foot, in the shank's frame
        std::string toolBody;                   // the MuJoCo body of that tool frame, to compare FK with the simulation
        MotionSafetyGate gate;
    };

    QuadrupedClient();
    ~QuadrupedClient();

    void setup(const std::array<RobotState*, kLegs>& legs) { _state = legs; }
    bool connect(const Config& cfg);            // realtime: returns at once, watch isConnected(); otherwise loads here and returns the result
    void update();
    void close();

    bool isConnected() const { return _connected.load(); }
    bool consumeJustConnected() { return _justConnected.exchange(false); }

    void setStreaming(bool on);
    bool isStreaming() const { return _streaming.load(); }
    void setJointLimits(int leg, const JointVec& lo, const JointVec& hi);
    uint16_t clampedMask() const { return _clampedMask.load(); }   // bit 3*leg + joint: that command was clamped to the limits
    bool speedLimited() const { return _speedLimited.load(); }
    double largestStepDeg() const { return _largestStep.load(); }   // the biggest change of one joint's command in one 5 ms cycle so far
    bool hasFault() const { return _hasFault.load(); }
    std::string faultMessage() const;
    void clearFault();                          // an operator action: it does not resume streaming
    std::string status() const;

    // The body.
    Sense sense() const;
    double torqueMax() const { return _torqueMax.load(); }          // the largest joint torque of the last step, N m
    double actuatorMicros() const { return _actuatorMicros.load(); }
    void resetRobot();                          // stand it up again at the spawn point (a request, carried out by the worker)
    void push(float vx, float vy);              // a shove, m/s in the body frame (a request)
    void setPaused(bool p) { _paused = p; }
    bool paused() const { return _paused.load(); }
    int resets() const { return _resets.load(); }                   // how many times it has been stood up again (a fall, or resetRobot())
    int falls() const { return _falls.load(); }

    // Real time off: one 5 ms physics step on the calling thread.
    void stepOffline();

    // The model (immutable once connected) and a consistent copy of the poses of the last step, for drawing.
    const mjModel* model() const { return _m; }
    void bodyPoses(std::vector<glm::mat4>& out) const;
    void geomPoses(std::vector<glm::mat4>& out) const;

    // ofxRobot's forward kinematics of a leg's foot, in the trunk frame, and where the simulation has it: they must agree (they
    // come from the same model), which is the check that the descriptors are right.
    const std::vector<Leg>& legs() const { return _legs; }
    glm::dmat4 footFk(int leg, const JointVec& q) const;
    glm::dvec3 toolFk(int leg) const;
    glm::dvec3 toolSim(int leg) const;
    double maxFkErrorMm() const;

    // The actuator network on its own, for the parity check: x = [n][pos error, velocity]; h, c = [2][n][8] carried between calls;
    // the torque out, before the motor clipping.
    bool runActuator(int n, std::vector<float>& x, std::vector<float>& h, std::vector<float>& c, std::vector<float>& torque);

    static constexpr double kDt = 0.005;       // the physics step, s; the policy period is four of them

private:
    void load();                                // the model, the network, the legs: the worker's first job (or connect()'s)
    void workerLoop();
    void cycle();                               // one 5 ms step: command, limits, gate, actuator network, mj_step, snapshot
    void resetNow();
    void publish();                             // the snapshot the main thread reads
    void fault(const std::string& why);
    std::array<float, kJoints> holdPoseDeg() const;

    Config _cfg;
    std::array<RobotState*, kLegs> _state{};
    mjModel* _m = nullptr;
    mjData* _d = nullptr;
    int _trunk = -1;
    std::vector<Leg> _legs;
    std::vector<int> _qpos, _dof, _actuator;    // MuJoCo addresses in the policy's joint order
    std::vector<float> _defaultRad;
    struct Motor { float saturation = 120.f, limit = 80.f, velocity = 7.5f, velAtLimit = 12.5f; } _motor;

    ofxOnnx _actNet;
    std::vector<const char*> _actIn, _actOut;
    std::vector<int> _actInIdx, _actOutIdx;
    std::vector<float> _ax, _ah, _ac, _atau;
    std::atomic<double> _actuatorMicros{0.0}, _torqueMax{0.0}, _largestStep{0.0};

    // The worker's own state.
    std::array<float, kJoints> _cmdDeg{};       // the command in force, deg
    std::array<float, kJoints> _limLoDeg{}, _limHiDeg{};
    mutable std::mutex _limMtx;
    int _fallCount = 0;

    std::thread _worker;
    std::atomic<bool> _running{false}, _ready{false}, _connected{false}, _justConnected{false};
    std::atomic<bool> _streaming{false}, _speedLimited{false}, _hasFault{false}, _paused{false};
    std::atomic<bool> _resetRequested{false}, _clearRequested{false};
    std::atomic<uint16_t> _clampedMask{0};
    std::atomic<int> _resets{0}, _falls{0};

    // Exchanged under _mtx.
    mutable std::mutex _mtx;
    std::array<JointVec, kLegs> _inTarget;
    bool _inHasTarget = false;
    glm::dvec3 _pushIn{0.0};
    bool _pushRequested = false;
    struct Snapshot {
        Sense sense;
        std::array<float, kJoints> cmdDeg{};
        std::vector<glm::mat4> bodyPose, geomPose;
        std::array<glm::dvec3, kLegs> footSim{};   // in the trunk frame
        long steps = 0, streamed = 0;
    } _snap;
    std::string _status = "idle", _faultMessage;
};
