#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"
#include "QuadrupedClient.h"

// The walking policy: the app's side of the loop, the part that decides where the legs go. Isaac Lab's pretrained
// Isaac-Velocity-Flat-AnymalD (a 3 x 128 ELU network: 48 numbers in, 12 joint position targets out), run through ofxOnnx.
//
// It does what an app does with an arm: reads what the controller reports (the legs' RobotStates, and for the body the client's
// sense(), which a real robot would get from its IMU and state estimator) and writes joint targets into RobotState::targetJoints
// (degrees, one RobotState per leg) for the client to send, through its gate, limits and speed cap, to the actuators.
//
// The policy is asked once per policy period (20 ms) of the controller's own time, so whatever drives the world (a worker thread in
// real time, or a recording stepping it frame by frame) the brain is asked at the same moments.
//
//   obs (48, no scaling): base linear velocity (body frame), base angular velocity, projected gravity, the velocity command
//   (vx, vy, wz), joint positions minus the default pose, joint velocities, the last action
//   action (12): the targets are default + 0.5 * action
class QuadrupedBrain {
public:
    static constexpr int kObs = 48, kJoints = QuadrupedClient::kJoints;

    bool setup(const std::string& dir = "anymal_d");
    void reset();                               // forget the last action and the heading to hold (after a fall, or a reset)

    // The velocity command, in the body frame: forward m/s, sideways m/s, turn rad/s. Clamped to what it was trained on.
    void setCommand(float vx, float vy, float wz);
    glm::vec3 command() const { return _cmd; }
    // What the policy is actually given: the same, except that with no turn commanded the turn rate comes from a heading
    // controller that holds the heading the robot had when the turn stopped (Isaac Lab's velocity command with heading_command
    // on, stiffness 0.5, which is how the policy was trained: it never saw a joystick turn rate). Without it the robot walks
    // straight to within a few degrees but drifts off line over ten metres.
    glm::vec3 appliedCommand() const { return _applied; }
    bool headingHold = true;

    // Once per frame (or per physics step), main thread: if a new policy period has begun in the controller's time, ask the
    // policy and write its 12 targets into the legs' RobotStates (hasTarget true). Returns how many times it asked.
    int tick(const QuadrupedClient::Sense& sense, const std::array<RobotState*, QuadrupedClient::kLegs>& legs);

    bool runPolicy(const std::vector<float>& obs, std::vector<float>& act);   // the network on its own (the parity check)
    double policyMicros() const { return _policyMicros; }
    const std::vector<float>& lastAction() const { return _last; }

private:
    ofxOnnx _policy;
    std::vector<float> _default;                // rad, in the policy's joint order
    float _actionScale = 0.5f;
    glm::vec2 _rangeX{-1.f, 1.f}, _rangeY{-1.f, 1.f}, _rangeZ{-1.f, 1.f};
    glm::vec3 _cmd{0.f}, _applied{0.f};
    double _headingTarget = 0.0;
    static constexpr float kHeadingStiffness = 0.5f;   // 1/s, from the training configuration
    static constexpr double kPeriod = 0.02;            // s: 50 Hz
    long _lastPeriod = -1;
    bool _needHeading = true;
    std::vector<float> _obs, _act, _last;
    double _policyMicros = 0.0;
};
