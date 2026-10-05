#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"
#include "ArmIk.h"                 // ofxRobot: forward kinematics of a limb
#include "MotionSafetyGate.h"      // ofxRobot: sanity check on the joint targets that get sent
#include "RobotKinematics.h"       // ofxRobot: the limbs, loaded from JSON descriptors
#include <mujoco/mujoco.h>         // ofxRobot/libs/mujoco
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

// The microduck biped, walking: three libraries doing three jobs.
//
//   MuJoCo      the physics: the robot's bodies, joints, servos (position actuators) and the floor
//   ofxOnnx     the brain: a walking policy, 136 numbers in (IMUs, joints, time-of-flight, command), 14 joint targets out, 50 times a second
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

    // Two threads: the physics (MuJoCo stepping in real time, 200 Hz, which owns the robot's state) and the inference (the
    // policy, which waits for an observation and answers with 14 targets). Every 50 Hz period the physics thread hands the
    // inference thread the observation and waits a moment for the answer, so the targets reach the servos as they did in
    // training, while neither thread waits for a frame to be drawn. While they run, hold `mutex` whenever you touch the
    // robot from the main thread: to read its state or draw it, and to call setCommand / reset / push / clearFault / paused.
    void startThread();
    void stopThread();
    std::mutex mutex;

    // `dir` is the folder under bin/data that holds microduck.xml, microduck.json, the limb descriptors and walker.onnx.
    // `skills` picks which policies to load, as bits of Skill: a controllable robot loads Walk and Run, a follower Follow.
    bool setup(const std::string& dir, unsigned skills = (1u << 0) | (1u << 1));
    // Where it stands after reset(): a position on the floor and the way it faces. Robots in separate Microduck objects
    // are separate MuJoCo worlds that share one coordinate frame, so they pass through each other instead of colliding.
    void setSpawn(double x, double y, double yawRadians) { _spawn = glm::dvec3(x, y, 0.0); _spawnYaw = yawRadians; }
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

    // A point lookDistance metres ahead of the trunk along its heading, at eye level, moving with the robot and swaying
    // left and right. It is only drawn: the Walk policy has no look-at input (its target slot is zero; the separate Look
    // skill is the one that takes a target).
    glm::dvec3 lookTarget() const;
    float lookDistance = 2.f;
    float lookSwayMetres = 0.05f;               // it shifts this far left and right of straight ahead...
    float lookSwaySeconds = 4.f;                // ...once per this many seconds

    // Advance by `wallSeconds` of real time: runs as many 50 Hz policy steps as that covers (each is four physics steps).
    void update(double wallSeconds);
    // One policy step: observe, ask the policy, send the targets, step the physics.
    void policyStep();

    // The two policies it can run, one at a time: Walk follows the velocity command; Run runs to a target on the floor and
    // stops there (the target is a point in the world, given to the policy in the robot's own frame).
    // Follow keeps a distance (0.6 m in training) behind another robot, facing it: the leader's position is the target.
    enum Skill { Walk = 0, Run = 1, Follow = 2 };
    void follow(const Microduck* leader) { _leader = leader; setSkill(Follow); }
    const Microduck* leader() const { return _leader; }
    glm::dvec3 publishedPos() const;                        // the trunk position as of the last step; safe from another robot's thread
    void setSkill(Skill s);
    Skill skill() const { return _skill; }
    bool hasSkill(Skill s) const { return _hasSkill[s]; }
    static const char* skillName(Skill s) { return s == Walk ? "walk" : s == Run ? "run" : "follow"; }
    glm::dvec3 target() const { return _target; }
    void placeTarget(double ahead, double left);           // relative to the robot's heading, on the floor
    void nudgeTarget(double ahead, double left);           // the same, added to where it is now
    double targetDistance() const;                         // along the floor, from the trunk

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
    void observe();                             // the heading controller and the observation
    void actuateAndStep(bool actionOk);         // the targets from _act through ofxRobot's gate to the servos, then four physics steps
    bool infer(const std::vector<float>& obs, std::vector<float>& act, Skill skill);   // the policy; one thread at a time
    void publishPose();                         // the position other robots read
    void physicsLoop();
    void inferenceLoop();
    void buildObservation();

    mjModel* _m = nullptr;
    mjData* _d = nullptr;
    int _trunk = -1;
    // The sensors of the real robot that the policy reads: two IMUs (sensor addresses into mjData::sensordata), the head's
    // body, and the time-of-flight sensor (a grid of rays out of the trunk).
    int _accBody = -1, _gyroBody = -1, _accHead = -1, _gyroHead = -1, _headBody = -1;
    int _numSkills = 6;
    struct Tof { glm::dvec3 pos{0.037, 0.0, 0.02}; double pitchUp = 15.0, fov = 60.0, maxDistance = 1.5, minDistance = 0.03; int zones = 8; } _tof;
    std::vector<glm::dvec3> _tofDirs;           // the rays, in the trunk frame, v-major like Isaac Lab's lidar pattern
    double _eyeHeight = 0.0;                    // eye level above the trunk origin in the default pose

    ofxOnnx _onnx[3];                           // one per skill
    bool _hasSkill[3] = {false, false, false};
    int _skillIndex[3] = {2, 3, 4};                // its slot in the observation's skill one-hot
    Skill _skill = Walk, _obsSkill = Walk, _mailSkill = Walk;
    glm::dvec3 _target{0.0};
    glm::dvec3 _spawn{0.0};
    double _spawnYaw = 0.0;
    const Microduck* _leader = nullptr;
    mutable std::mutex _poseMutex;
    glm::dvec3 _pose{0.0};
    std::vector<float> _obs, _act, _last;       // 136 in, 14 out, and the last action (an input)
    std::atomic<double> _policyMicros{0.0};

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

    std::thread _physicsThread, _inferenceThread;
    std::atomic<bool> _run{false};
    // The hand-off between them: the observation going over, the action coming back, each tagged with the period it is for.
    std::mutex _mail;
    std::condition_variable _obsReady, _actReady;
    std::vector<float> _mailObs, _mailAct;
    long _obsSeq = 0, _actSeq = 0, _inferSeq = 0;
    bool _mailOk = false;

    std::vector<Limb> _limbs;
    bool _faulted = false;
    std::string _faultReason;
};
