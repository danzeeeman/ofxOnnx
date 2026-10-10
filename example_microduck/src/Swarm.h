#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"
#include "MotionSafetyGate.h"      // ofxRobot: sanity check on the joint targets that get sent
#include <mujoco/mujoco.h>         // ofxRobot/libs/mujoco
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

// A swarm of microducks in ONE MuJoCo world, inside a walled arena (Isaac Lab's enclosed navigation task, with more ducks).
//
// They share a world so that each duck's time-of-flight sensor, a grid of rays out of its chest, sees the other ducks and the
// walls, and so that they really collide. Every duck runs the same policy (the NavigateWalkEnclosed one, which steers by
// what its time-of-flight sensor sees), commanded to walk forwards at its own pace with a heading that wanders; what keeps
// them from hitting each other is the policy.
//
// Two threads: the physics (one MuJoCo world, stepped in real time) and the inference (the policy, once per duck). Every
// 50 Hz period the physics thread builds every duck's observation, hands them over, and waits a moment for the actions.
// ofxRobot's MotionSafetyGate checks every duck's joint targets (three limbs each) before they reach the servos.
class Swarm {
public:
    ~Swarm();

    // `dir` is the folder under bin/data with swarm.xml, microduck.json and navigator.onnx (see tools/make_swarm.py).
    bool setup(const std::string& dir);
    void reset();

    void startThread();
    void stopThread();
    // One 50 Hz period for every duck on this thread (the headless check).
    void policyStep();

    std::mutex mutex;               // hold it from the main thread to read or draw while the threads run
    bool paused = false;

    int count() const { return (int)_ducks.size(); }
    const mjModel* model() const { return _m; }
    const mjData* data() const { return _d; }
    double simTime() const { return _d ? _d->time : 0.0; }
    double policyMicros() const { return _policyMicros; }       // for all the ducks of a period
    glm::dvec3 pos(int k) const;
    double heading(int k) const;
    double uprightness(int k) const;
    bool faulted(int k) const { return _ducks[k].faulted; }
    int duckOfBody(int body) const { return _bodyDuck[body]; }
    float command(int k) const { return _ducks[k].vx; }
    int raysOnDucks(int k) const { return _ducks[k].raysOnDucks; }
    int raysOnWalls(int k) const { return _ducks[k].raysOnWalls; }

    // What has happened since reset(): contacts between two ducks, and between a duck and a wall, counted in policy periods
    // that had one (a duck leaning on another for a second counts 50), and how close two ducks have been.
    struct Stats {
        long duckContactPeriods = 0, wallContactPeriods = 0, obstacleContactPeriods = 0;
        long duckPairs = 0;                     // distinct pairs of ducks that have touched
        double closest = 1e9;                   // between two trunks, in metres
    };
    Stats stats() const { return _stats; }
    // where a period's time goes, averaged since reset(), in microseconds: observations (the time-of-flight rays), the policy, and
    // the servos plus four physics steps plus the contact scan
    double microsObserve() const { return _periods ? _tObserve / _periods : 0.0; }
    double microsInfer() const { return _periods ? _tInfer / _periods : 0.0; }
    double microsStep() const { return _periods ? _tStep / _periods : 0.0; }

private:
    struct Duck {
        int trunk = -1, head = -1, freeQpos = -1;
        std::vector<int> obsQpos, obsDof, actuator;
        int accBody = -1, gyroBody = -1, accHead = -1, gyroHead = -1;
        std::vector<float> obs, act, last;
        float vx = 0.3f;
        glm::vec3 applied{0.f};
        double headingTarget = 0.0, nextHeading = 0.0;
        bool faulted = false;
        int downPeriods = 0;                      // how long it has been on its side
        int raysOnDucks = 0, raysOnWalls = 0;     // of its 64 time-of-flight rays, how many hit another duck / a wall
        struct Limb { std::vector<int> actionIdx; MotionSafetyGate gate; JointVec target; };
        std::vector<Limb> limbs;
    };

    void observe();
    void buildObservation(Duck& d);
    bool inferAll();
    void actuateAndStep(bool actionsOk);
    void physicsLoop();
    void inferenceLoop();
    void scanContacts();
    void respawn(int k);                        // stand a fallen duck up at a free spot
    glm::dvec2 freeSpot(int exceptDuck);

    mjModel* _m = nullptr;
    mjData* _d = nullptr;
    std::vector<Duck> _ducks;
    std::vector<int> _bodyDuck;                 // the duck a body belongs to, or -1 (the floor, the walls)
    std::vector<int> _geomDuck;                 // the same for geoms; the arena's walls are -2, the obstacles -3
    std::vector<glm::dvec3> _obstacles;         // x, y, and the radius of the circle around each, for placing ducks clear of them
    bool clearOfObstacles(const glm::dvec2& p, double margin) const;
    std::vector<std::vector<char>> _touched;    // pairs that have touched
    Stats _stats;
    double _tObserve = 0, _tInfer = 0, _tStep = 0;
    long _periods = 0;

    ofxOnnx _onnx;
    std::atomic<double> _policyMicros{0.0};
    std::vector<float> _defaultAct, _defaultObs, _scale;
    int _decimation = 4;
    double _simDt = 0.005, _initHeight = 0.119, _half = 2.0;
    struct Tof { glm::dvec3 pos{0.037, 0.0, 0.02}; double pitchUp = 15.0, fov = 60.0, maxDistance = 1.5, minDistance = 0.03; int zones = 8; } _tof;
    std::vector<glm::dvec3> _tofDirs;
    int _skillIndex = 2, _numSkills = 6;
    static constexpr float kHeadingStiffness = 0.5f;
    std::mt19937 _rng{2024};

    std::thread _physicsThread, _inferenceThread;
    std::atomic<bool> _run{false};
    std::mutex _mail;
    std::condition_variable _obsReady, _actReady;
    std::vector<float> _mailObs, _mailAct;
    long _obsSeq = 0, _actSeq = 0, _inferSeq = 0;
    bool _mailOk = false;
    std::vector<float> _allObs, _allAct;        // the physics thread's copies: every duck's, back to back
    bool _actionsFresh = false;
};
