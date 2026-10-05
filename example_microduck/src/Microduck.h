#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"
#include "ArmIk.h"                 // ofxRobot: forward kinematics of a limb
#include "MotionSafetyGate.h"      // ofxRobot: sanity check on the joint targets that get sent
#include "RobotKinematics.h"       // ofxRobot: the limbs, loaded from JSON descriptors
#include <mujoco/mujoco.h>         // ofxRobot/libs/mujoco
#include <atomic>
#include <mutex>
#include <thread>

// The microduck biped, walking: three libraries doing three jobs.
//
//   MuJoCo      the physics: the robot's bodies, joints, servos (position actuators) and the floor
//   ofxOnnx     the brain: a walking policy, 57 numbers in, 14 joint targets out, 50 times a second
//   ofxRobot    the robot as data: each limb (two legs, the head) is a serial chain loaded from a JSON descriptor
//               (RobotKinematics), whose forward kinematics (ArmIk) gives the joint and foot positions; and a
//               MotionSafetyGate per limb checks every set of joint targets before it reaches the servos
//
// The policy was trained in Isaac Lab (Isaac-Velocity-Flat-Microduck); this reproduces its physics and its
// observation exactly, so it walks here as it did there. tools/make_microduck.py makes the data this loads.
class Microduck {
public:
    // One limb, as ofxRobot sees it.
    struct Limb {
        std::string name;                       // left_leg, right_leg, head
        RobotKinematics kin;                    // from <name>.joints.json
        ArmIk ik;                               // forward kinematics from the same chain
        glm::dmat4 tool{1.0};                   // the foot (or the mouth tip) in the last link's frame
        std::vector<int> actionIdx;             // where this limb's joints sit in the 14 action values
        std::vector<int> qposAdr;               // and in MuJoCo's qpos
        std::string toolBody;                   // the MuJoCo body of that tool frame, to compare FK with the simulation
        MotionSafetyGate gate;
        JointVec q;                             // current joint angles, degrees
        JointVec target;                        // the targets last sent, degrees
    };

    ~Microduck();

    // The simulation and the policy can run on their own thread, 50 Hz in real time, so inference never waits for a
    // frame to be drawn (or the other way round). While it runs, hold `mutex` whenever you touch the robot from the
    // main thread: to read its state or draw it, and to call setCommand / reset / push / clearFault / paused.
    void startThread();
    void stopThread();
    std::mutex mutex;

    // `dir` is the folder under bin/data that holds microduck.xml, microduck.json, the limb descriptors and walker.onnx.
    bool setup(const std::string& dir);
    void reset();

    // The velocity command, in the body frame: forward m/s, sideways m/s, turn rad/s. Clamped to what it was trained on.
    void setCommand(float vx, float vy, float wz);
    glm::vec3 command() const { return _cmd; }
    // What the policy is actually given: the same, except that with no turn commanded the turn rate comes from a heading
    // controller that holds the heading the robot had when the turn stopped. The policy was trained that way (Isaac Lab's
    // velocity command with heading_command on, stiffness 0.5): it never saw a joystick turn rate, and without this it
    // walks in a circle.
    glm::vec3 appliedCommand() const { return _applied; }
    double heading() const;                     // yaw, radians

    // The point the head is asked to look at: lookDistance metres ahead of the trunk along its heading, at eye level,
    // moving with the robot (Isaac Lab attaches it to the yaw-aligned root frame, so roll and pitch do not move it).
    // The policy was trained with it 0.4-1.2 m away; further than that is outside what it saw.
    glm::dvec3 lookTarget() const;
    float lookDistance = 2.f;
    float lookSwayMetres = 0.05f;               // it shifts this far left and right of straight ahead...
    float lookSwaySeconds = 4.f;                // ...once per this many seconds

    // Advance by `wallSeconds` of real time: runs as many 50 Hz policy steps as that covers (each is four physics steps).
    void update(double wallSeconds);
    // One policy step: observe, ask the policy, send the targets, step the physics.
    void policyStep();

    // A shove, in m/s sideways, to see it recover.
    void push(float vx, float vy);

    // State.
    const mjModel* model() const { return _m; }
    const mjData* data() const { return _d; }
    double simTime() const { return _d ? _d->time : 0.0; }
    glm::dvec3 trunkPos() const;
    glm::dmat3 trunkRot() const;                // body to world
    glm::dvec3 velocityBody() const;            // linear velocity in the body frame
    double uprightness() const;                 // 1 = upright, 0 = on its side, -1 = upside down
    double policyMicros() const { return _policyMicros; }
    bool faulted() const { return _faulted; }
    const std::string& faultReason() const { return _faultReason; }
    void clearFault();
    std::vector<Limb>& limbs() { return _limbs; }
    const std::vector<Limb>& limbs() const { return _limbs; }

    // ofxRobot's forward kinematics of a limb's tool, in the trunk frame, and where the simulation has it: they
    // must agree (they come from the same URDF), which is the check that the descriptors are right.
    glm::dvec3 toolFk(const Limb& l) const;
    glm::dvec3 toolSim(const Limb& l) const;
    double maxFkErrorMm() const;

    bool paused = false;

private:
    void buildObservation();
    void applyTargets(const std::vector<float>& action);

    mjModel* _m = nullptr;
    mjData* _d = nullptr;
    int _trunk = -1;
    double _eyeHeight = 0.0;                    // eye level above the trunk origin in the default pose

    ofxOnnx _onnx;
    std::vector<float> _obs, _act, _last;       // 57 in, 14 out, and the last action (an input)
    double _policyMicros = 0.0;

    // From microduck.json.
    std::vector<std::string> _jointOrder, _actionOrder;
    std::vector<int> _obsQpos, _obsDof;         // MuJoCo addresses in the observation's joint order
    std::vector<int> _actuator;                 // the actuator of each action
    std::vector<float> _defaultObs, _defaultAct, _scale;
    float _initHeight = 0.117f;
    int _decimation = 4;
    double _simDt = 0.005;
    glm::vec2 _rangeX{-0.2f, 0.4f}, _rangeY{-0.2f, 0.2f}, _rangeZ{-1.f, 1.f};

    glm::vec3 _cmd{0.f}, _applied{0.f};
    double _headingTarget = 0.0;
    static constexpr float kHeadingStiffness = 0.5f;   // 1/s, from the training configuration
    double _accum = 0.0;

    std::thread _thread;
    std::atomic<bool> _run{false};

    std::vector<Limb> _limbs;
    bool _faulted = false;
    std::string _faultReason;
};
