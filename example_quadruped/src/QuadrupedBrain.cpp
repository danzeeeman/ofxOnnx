#include "QuadrupedBrain.h"
#include <chrono>

namespace {
const double kDeg = 180.0 / 3.14159265358979;
}

bool QuadrupedBrain::setup(const std::string& dir) {
    if (!ofFile::doesFileExist(dir + "/anymal_d.json") || !ofFile::doesFileExist(dir + "/policy.onnx")) {
        ofLogError("QuadrupedBrain") << "no policy in " << ofToDataPath(dir, true);
        return false;
    }
    const ofJson meta = ofLoadJson(dir + "/anymal_d.json");
    _actionScale = meta["action"]["scale"].get<float>();
    for (const std::string& n : meta["joint_order"].get<std::vector<std::string>>()) _default.push_back(meta["default_joint_pos_rad"][n].get<float>());
    if ((int)_default.size() != kJoints) { ofLogError("QuadrupedBrain") << "expected 12 joints"; return false; }
    const auto& r = meta["command_range"];
    _rangeX = {r["vx"][0].get<float>(), r["vx"][1].get<float>()};
    _rangeY = {r["vy"][0].get<float>(), r["vy"][1].get<float>()};
    _rangeZ = {r["wz"][0].get<float>(), r["wz"][1].get<float>()};

    ofxOnnx::Settings settings;
    settings.numIntraOpThreads = 1;           // a tiny network: threads would only add cost
    settings.logLevel = OF_LOG_WARNING;
    settings.bPrintModelInfo = false;
    settings.modelPath = dir + "/policy.onnx";
    if (!_policy.load(settings)) return false;
    _obs.assign(kObs, 0.f);
    _act.assign(kJoints, 0.f);
    _last.assign(kJoints, 0.f);
    ofLogNotice("QuadrupedBrain") << "policy " << settings.modelPath;
    return true;
}

void QuadrupedBrain::reset() {
    std::fill(_last.begin(), _last.end(), 0.f);
    std::fill(_act.begin(), _act.end(), 0.f);
    _lastPeriod = -1;
    _headingTarget = 0.0;
    _applied = _cmd;
    _needHeading = true;
}

void QuadrupedBrain::setCommand(float vx, float vy, float wz) {
    _cmd = glm::vec3(glm::clamp(vx, _rangeX.x, _rangeX.y), glm::clamp(vy, _rangeY.x, _rangeY.y), glm::clamp(wz, _rangeZ.x, _rangeZ.y));
}

bool QuadrupedBrain::runPolicy(const std::vector<float>& obs, std::vector<float>& act) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = _policy.runFloat(obs, act) && (int)act.size() == kJoints;
    _policyMicros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

int QuadrupedBrain::tick(const QuadrupedClient::Sense& s, const std::array<RobotState*, QuadrupedClient::kLegs>& legs) {
    if (!s.valid) return 0;
    const long period = (long)std::floor(s.simTime / kPeriod + 1e-6);
    if (period == _lastPeriod) return 0;
    _lastPeriod = period;

    // The heading to hold: the heading the robot has when no turn is being commanded.
    const double yaw = std::atan2(s.rot[0][1], s.rot[0][0]);
    if (_needHeading) { _headingTarget = yaw; _needHeading = false; }
    _applied = _cmd;
    if (headingHold) {
        if (std::fabs(_cmd.z) > 1e-3f) {
            _headingTarget = yaw;                                    // turning: the turn rate is yours
        } else {
            double err = _headingTarget - yaw;
            err = std::atan2(std::sin(err), std::cos(err));          // wrapped to (-pi, pi]
            _applied.z = glm::clamp(kHeadingStiffness * (float)err, _rangeZ.x, _rangeZ.y);
        }
    }

    // The policy's 48 inputs, in the order it was trained with (no scaling, no normalisation).
    size_t k = 0;
    auto put = [&](const glm::dvec3& v) { for (int i = 0; i < 3; ++i) _obs[k++] = (float)v[i]; };
    put(s.linVelBody);                                // base_lin_vel
    put(s.angVelBody);                                // base_ang_vel
    put(s.gravityBody);                               // projected_gravity
    for (int i = 0; i < 3; ++i) _obs[k++] = _applied[i];                              // velocity_commands: vx, vy, wz
    for (int i = 0; i < kJoints; ++i) _obs[k++] = s.q[i] - _default[i];               // joint_pos_rel
    for (int i = 0; i < kJoints; ++i) _obs[k++] = s.qd[i];                            // joint_vel_rel
    for (int i = 0; i < kJoints; ++i) _obs[k++] = _last[i];                           // last_action

    if (!runPolicy(_obs, _act)) { ofLogError("QuadrupedBrain") << "the policy did not answer"; return 0; }
    _last = _act;

    // The targets: the default pose plus 0.5 * the action, in degrees, into each leg's RobotState.
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) {
        JointVec t(QuadrupedClient::kLegJoints);
        for (int j = 0; j < QuadrupedClient::kLegJoints; ++j) {
            const int i = l * QuadrupedClient::kLegJoints + j;
            t[j] = (_default[i] + _actionScale * _act[i]) * kDeg;
        }
        legs[l]->targetJoints = t;
        legs[l]->hasTarget = true;
    }
    return 1;
}
