#include "QuadrupedClient.h"
#include <chrono>

namespace {
glm::dvec3 col3(const mjtNum* v) { return glm::dvec3(v[0], v[1], v[2]); }
const double kRad = 3.14159265358979 / 180.0;
}

QuadrupedClient::QuadrupedClient() {
    for (auto& t : _inTarget) t = JointVec(kLegJoints);
}

QuadrupedClient::~QuadrupedClient() {
    close();
    if (_d) mj_deleteData(_d);
    if (_m) mj_deleteModel(_m);
}

void QuadrupedClient::setStreaming(bool on) {
    if (on && _hasFault.load()) {
        ofLogWarning("QuadrupedClient") << "streaming refused: fault (" << faultMessage() << "), clear it first";
        return;
    }
    if (on != _streaming.exchange(on)) ofLogNotice("QuadrupedClient") << "streaming " << (on ? "ON" : "off (holding the stand pose)");
}

void QuadrupedClient::setJointLimits(int leg, const JointVec& lo, const JointVec& hi) {
    std::lock_guard<std::mutex> lk(_limMtx);
    for (int j = 0; j < kLegJoints && j < lo.size() && j < hi.size(); ++j) {
        _limLoDeg[leg * kLegJoints + j] = (float)lo[j];
        _limHiDeg[leg * kLegJoints + j] = (float)hi[j];
    }
}

std::string QuadrupedClient::faultMessage() const {
    std::lock_guard<std::mutex> lk(_mtx);
    return _faultMessage;
}

std::string QuadrupedClient::status() const {
    std::lock_guard<std::mutex> lk(_mtx);
    return _status;
}

void QuadrupedClient::fault(const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _faultMessage = why;
        _status = "FAULT: " + why;
    }
    _hasFault = true;
    _streaming = false;      // a fault stops streaming at once; it stays off until the operator clears it and turns it on again
    ofLogError("QuadrupedClient") << "fault: " << why << " (streaming stopped, holding the stand pose)";
}

// An operator action, carried out by the worker at its next cycle (the gates belong to it).
void QuadrupedClient::clearFault() { _clearRequested = true; }

bool QuadrupedClient::connect(const Config& cfg) {
    close();
    _cfg = cfg;
    if (!cfg.realtime) {                         // no worker: the caller steps it
        load();
        if (!_ready.load()) return false;
        _connected = true;
        _justConnected = true;
        return true;
    }
    _running = true;
    _worker = std::thread(&QuadrupedClient::workerLoop, this);
    return true;
}

void QuadrupedClient::close() {
    if (_running.exchange(false)) {
        if (_worker.joinable()) _worker.join();
        ofLogNotice("QuadrupedClient") << "quadruped controller stopped";
    } else if (_worker.joinable()) {
        _worker.join();   // the worker ended itself (the model did not load)
    }
    _connected = false;
    _streaming = false;
    for (RobotState* s : _state) if (s) s->egmConnected = false;
}

void QuadrupedClient::resetRobot() { _resetRequested = true; }

void QuadrupedClient::push(float vx, float vy) {
    std::lock_guard<std::mutex> lk(_mtx);
    _pushIn = glm::dvec3(vx, vy, 0.0);
    _pushRequested = true;
}

void QuadrupedClient::workerLoop() {
    load();
    if (!_ready.load()) { _running = false; return; }
    _connected = true;
    _justConnected = true;
    using clock = std::chrono::steady_clock;
    auto last = clock::now();
    double accum = 0.0;
    while (_running.load()) {
        const auto now = clock::now();
        accum += std::min(std::chrono::duration<double>(now - last).count(), 0.1);   // after a long stall, do not try to catch up all of it
        last = now;
        while (accum >= kDt && _running.load()) {
            accum -= kDt;
            if (!_paused.load()) cycle();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void QuadrupedClient::stepOffline() { cycle(); }

// The model, the network, the legs.
void QuadrupedClient::load() {
    const std::string dir = _cfg.dir, base = ofToDataPath(dir, true) + "/";
    auto fail = [&](const std::string& why) {
        ofLogError("QuadrupedClient") << why;
        std::lock_guard<std::mutex> lk(_mtx);
        _status = "failed: " + why;
    };
    if (!ofFile::doesFileExist(base + "anymal_d.xml") || !ofFile::doesFileExist(base + "anymal_d.json")) {
        fail("no model in " + base + ": see the README (tools/make_anymal_d.py, make_limbs.py, export_models.py)");
        return;
    }
    const ofJson meta = ofLoadJson(dir + "/anymal_d.json");

    // 1. Physics: the model, with the floor in it and one motor per joint (the torque comes from the actuator network).
    char error[1000] = "";
    _m = mj_loadXML((base + "anymal_d.xml").c_str(), nullptr, error, sizeof(error));
    if (!_m) { fail(std::string("MuJoCo could not load the model: ") + error); return; }
    _d = mj_makeData(_m);
    _trunk = mj_name2id(_m, mjOBJ_BODY, meta["bodies"]["trunk"].get<std::string>().c_str());
    if (_trunk < 0) { fail("no trunk body"); return; }
    if (std::fabs(_m->opt.timestep - kDt) > 1e-9) ofLogWarning("QuadrupedClient") << "the model's time step " << _m->opt.timestep << " is not " << kDt;
    const auto& clip = meta["actuator"]["clip"];
    _motor.saturation = clip["saturation_effort"].get<float>();
    _motor.limit = clip["actuator_effort_limit"].get<float>();
    _motor.velocity = clip["actuator_velocity_limit"].get<float>();
    _motor.velAtLimit = clip["vel_at_effort_limit"].get<float>();

    // 2. The policy's joint order (the same as the model's: depth-first by leg, HAA HFE KFE) and its default pose.
    const std::vector<std::string> order = meta["joint_order"].get<std::vector<std::string>>();
    if ((int)order.size() != kJoints) { fail("expected 12 joints"); return; }
    for (const std::string& n : order) {
        const int j = mj_name2id(_m, mjOBJ_JOINT, n.c_str());
        const int a = mj_name2id(_m, mjOBJ_ACTUATOR, n.c_str());
        if (j < 0 || a < 0) { fail("no joint or motor " + n); return; }
        _qpos.push_back(_m->jnt_qposadr[j]);
        _dof.push_back(_m->jnt_dofadr[j]);
        _actuator.push_back(a);
        _defaultRad.push_back(meta["default_joint_pos_rad"][n].get<float>());
    }

    // 3. The actuator network.
    ofxOnnx::Settings settings;
    settings.numIntraOpThreads = 1;           // a tiny network: threads would only add cost
    settings.logLevel = OF_LOG_WARNING;
    settings.bPrintModelInfo = false;
    settings.modelPath = dir + "/" + meta["actuator"]["network"].get<std::string>();
    if (!_actNet.load(settings)) { fail("the actuator network did not load"); return; }
    if (_actNet.getInputCount() != 3 || _actNet.getOutputCount() != 3) { fail("the actuator network should have 3 inputs and 3 outputs"); return; }
    for (int i = 0; i < 3; ++i) {   // which model input is x, h, c and which output is torque, h_out, c_out
        const std::string in = _actNet.getInputName(i), out = _actNet.getOutputName(i);
        _actInIdx.push_back(in == "x" ? 0 : in == "h" ? 1 : in == "c" ? 2 : -1);
        _actOutIdx.push_back(out == "torque" ? 0 : out == "h_out" ? 1 : out == "c_out" ? 2 : -1);
    }
    _actIn = _actNet.getInputNames();
    _actOut = _actNet.getOutputNames();
    _ax.assign(kJoints * 2, 0.f); _ah.assign(2 * kJoints * 8, 0.f); _ac.assign(2 * kJoints * 8, 0.f); _atau.assign(kJoints, 0.f);

    // 4. The robot as ofxRobot sees it: one serial chain per leg, from a JSON descriptor, and its forward kinematics.
    _legs.resize(kLegs);
    int leg = 0;
    for (auto it = meta["limb_files"].begin(); it != meta["limb_files"].end() && leg < kLegs; ++it, ++leg) {
        Leg& l = _legs[leg];
        l.name = it.key();
        if (!l.kin.load(ofToDataPath(dir + "/" + it.value().get<std::string>(), true))) { fail(l.name + ": the descriptor did not load"); return; }
        if (l.kin.numJoints() != kLegJoints) { fail(l.name + ": expected 3 joints"); return; }
        std::vector<ArmIk::JointGeom> chain(l.kin.numJoints());
        for (int i = 0; i < l.kin.numJoints(); ++i) {
            const auto& j = l.kin.joint(i);
            chain[i].origin = glm::dvec3(j.origin);
            chain[i].rot0 = glm::dquat(j.rot0);
            chain[i].rot1 = glm::dquat(j.rot1);
            l.kin.limits(i, chain[i].lo, chain[i].hi);
            if (order[leg * kLegJoints + i] != j.name) { fail(l.name + ": the descriptor's joints are not in the policy's order"); return; }
            _limLoDeg[leg * kLegJoints + i] = (float)chain[i].lo;     // the joint limits start as the descriptor's
            _limHiDeg[leg * kLegJoints + i] = (float)chain[i].hi;
        }
        std::string why;
        if (!ArmIk::paramsFromChainFkOnly(chain, l.ik.params, why)) { fail(l.name + ": " + why); return; }

        // The tool frame (the foot): a fixed frame after the last joint.
        const ofJson desc = ofLoadJson(dir + "/" + it.value().get<std::string>());
        const auto& o = desc["tool"]["origin"];
        const auto& rpy = desc["tool"]["rpy"];
        l.tool = glm::translate(glm::dmat4(1.0), glm::dvec3(o[0].get<double>(), o[1].get<double>(), o[2].get<double>())) *
                 glm::mat4_cast(glm::dquat(glm::dvec3(rpy[0].get<double>(), rpy[1].get<double>(), rpy[2].get<double>())));
        l.toolBody = desc["tool"]["link"].get<std::string>();
        if (mj_name2id(_m, mjOBJ_BODY, l.toolBody.c_str()) < 0) { fail("no body " + l.toolBody); return; }

        // The targets come from a neural network by design in steps of tens of degrees every 20 ms, so the gate is set to catch
        // only what is wrong: NaNs, and jumps no leg could follow (200 degrees in one 5 ms cycle).
        l.gate.params.maxJumpDegPerSec = 40000.f;
    }
    if (leg != kLegs) { fail("expected 4 legs"); return; }

    resetNow();
    _resets = 0;
    _ready = true;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _status = "connected, holding";
    }
    ofLogNotice("QuadrupedClient") << "ready: " << _m->nbody << " bodies, " << kLegs << " legs (ofxRobot), actuator network " << settings.modelPath;
}

std::array<float, QuadrupedClient::kJoints> QuadrupedClient::holdPoseDeg() const {
    std::array<float, kJoints> h{};
    for (int i = 0; i < kJoints; ++i) h[i] = _defaultRad[i] / (float)kRad;
    return h;
}

// Standing at the default pose, 0.62 m up, at the spawn point; the actuator network starts from rest.
void QuadrupedClient::resetNow() {
    mj_resetData(_m, _d);
    const double c = std::cos(_cfg.spawnYaw / 2.0), s = std::sin(_cfg.spawnYaw / 2.0);
    _d->qpos[0] = _cfg.spawnX; _d->qpos[1] = _cfg.spawnY; _d->qpos[2] = 0.62;
    _d->qpos[3] = c; _d->qpos[4] = 0.0; _d->qpos[5] = 0.0; _d->qpos[6] = s;
    for (int i = 0; i < kJoints; ++i) {
        _d->qpos[_qpos[i]] = _defaultRad[i];
        _d->ctrl[_actuator[i]] = 0.0;
    }
    mj_forward(_m, _d);
    std::fill(_ah.begin(), _ah.end(), 0.f);
    std::fill(_ac.begin(), _ac.end(), 0.f);
    _cmdDeg = holdPoseDeg();
    for (Leg& l : _legs) l.gate.idle();
    _torqueMax = 0.0;
    ++_resets;
    publish();
}

// One 5 ms step: what the app asked for, checked and limited, through the actuator network into the physics.
void QuadrupedClient::cycle() {
    std::array<JointVec, kLegs> in;
    bool hasTarget = false, doPush = false;
    glm::dvec3 pushV(0.0);
    {
        std::lock_guard<std::mutex> lk(_mtx);
        in = _inTarget;
        hasTarget = _inHasTarget;
        doPush = _pushRequested;
        pushV = _pushIn;
        _pushRequested = false;
    }
    if (_resetRequested.exchange(false)) resetNow();
    if (_clearRequested.exchange(false)) {
        for (Leg& l : _legs) l.gate.clearFault();
        {
            std::lock_guard<std::mutex> lk(_mtx);
            _faultMessage.clear();
            _status = "connected, holding";
        }
        _hasFault = false;
        ofLogNotice("QuadrupedClient") << "fault cleared by the operator (streaming stays off until it is turned on)";
    }
    if (doPush) {
        const mjtNum* x = _d->xmat + 9 * _trunk;
        const glm::dvec3 w(x[0] * pushV.x + x[1] * pushV.y, x[3] * pushV.x + x[4] * pushV.y, 0.0);
        _d->qvel[0] += w.x;
        _d->qvel[1] += w.y;
    }

    // 1. The command. Streaming off (or no target yet, or a fault): the stand pose. Streaming on: the app's targets, through
    //    ofxRobot's gate (a fault stops streaming), the joint limits, and the speed cap.
    const std::array<float, kJoints> hold = holdPoseDeg();
    std::array<float, kJoints> want = hold;
    bool streaming = _streaming.load() && hasTarget && !_hasFault.load();
    if (streaming) {
        for (int l = 0; l < kLegs; ++l) {
            for (int j = 0; j < kLegJoints; ++j) want[l * kLegJoints + j] = (float)in[l][j];
            if (!_legs[l].gate.observe(in[l], (float)kDt)) {
                fault(_legs[l].name + ": " + _legs[l].gate.faultReason());
                streaming = false;
                break;
            }
        }
    } else {
        for (Leg& l : _legs) l.gate.idle();
    }
    uint16_t clamped = 0;
    bool limited = false;
    if (streaming) {
        std::lock_guard<std::mutex> lk(_limMtx);
        for (int i = 0; i < kJoints; ++i) {
            const float v = std::min(std::max(want[i], _limLoDeg[i]), _limHiDeg[i]);
            if (v != want[i]) clamped |= (uint16_t)(1u << i);
            want[i] = v;
        }
    } else {
        want = hold;
    }
    const double rate = streaming ? _cfg.maxJointDegPerSec : _cfg.holdDegPerSec;
    const float maxStep = (float)(rate * kDt);
    double biggest = _largestStep.load();
    for (int i = 0; i < kJoints; ++i) {
        const float d = want[i] - _cmdDeg[i];
        if (streaming) biggest = std::max(biggest, (double)std::fabs(d));
        if (std::fabs(d) > maxStep) { _cmdDeg[i] += d > 0.f ? maxStep : -maxStep; limited = true; }
        else _cmdDeg[i] = want[i];
    }
    _largestStep = biggest;
    _clampedMask = clamped;
    _speedLimited = streaming && limited;

    // 2. The actuator network: the torque from the position error and the joint velocity, kept inside the motor's torque-speed
    //    curve (Isaac Lab's DC motor limits); then the physics.
    for (int i = 0; i < kJoints; ++i) {
        _ax[2 * i] = _cmdDeg[i] * (float)kRad - (float)_d->qpos[_qpos[i]];
        _ax[2 * i + 1] = (float)_d->qvel[_dof[i]];
    }
    if (!runActuator(kJoints, _ax, _ah, _ac, _atau)) {
        fault("the actuator network failed");
        return;
    }
    double tmax = 0.0;
    for (int i = 0; i < kJoints; ++i) {
        const float v = std::min(std::max(_ax[2 * i + 1], -_motor.velAtLimit), _motor.velAtLimit);
        const float top = std::min(_motor.saturation * (1.f - v / _motor.velocity), _motor.limit);
        const float bottom = std::max(_motor.saturation * (-1.f - v / _motor.velocity), -_motor.limit);
        const float tau = std::min(std::max(_atau[i], bottom), top);
        _d->ctrl[_actuator[i]] = tau;
        tmax = std::max(tmax, (double)std::fabs(tau));
    }
    _torqueMax = tmax;
    mj_step(_m, _d);
    // mj_step integrates after it computes the bodies' poses, so xpos/xmat lag qpos by a step: refresh them, so that what is
    // read from them (the drawing, the forward-kinematics check) belongs to the same state as the joint angles.
    mj_kinematics(_m, _d);

    // 3. A robot that fell over (tilted more than 60 degrees, or lower than 0.25 m) is stood up again, and counted.
    if (!std::isfinite(_d->qpos[2])) { resetNow(); return; }
    if (_cfg.autoReset && (_d->xmat[9 * _trunk + 8] < 0.5 || _d->qpos[2] < 0.25)) { ++_falls; resetNow(); return; }
    publish();
    if (streaming) {
        std::lock_guard<std::mutex> lk(_mtx);
        ++_snap.streamed;
    }
}

// The snapshot the main thread reads: one consistent step, never a half-stepped state.
void QuadrupedClient::publish() {
    Sense s;
    s.valid = true;
    s.simTime = _d->time;
    s.pos = col3(_d->xpos + 3 * _trunk);
    const mjtNum* x = _d->xmat + 9 * _trunk;    // row-major, body to world
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) s.rot[c][r] = x[3 * r + c];   // glm is column-major
    const glm::dmat3 Rt = glm::transpose(s.rot);
    s.linVelBody = Rt * col3(_d->qvel);         // MuJoCo's free joint has the linear velocity in the world frame
    s.angVelBody = col3(_d->qvel + 3);          // and the angular velocity already in the body frame
    s.gravityBody = Rt * glm::dvec3(0.0, 0.0, -1.0);
    for (int i = 0; i < kJoints; ++i) {
        s.q[i] = (float)_d->qpos[_qpos[i]];
        s.qd[i] = (float)_d->qvel[_dof[i]];
    }
    std::lock_guard<std::mutex> lk(_mtx);
    _snap.sense = s;
    _snap.cmdDeg = _cmdDeg;
    _snap.bodyPose.resize(_m->nbody);
    _snap.geomPose.resize(_m->ngeom);
    auto mat = [](const mjtNum* pos, const mjtNum* m9) {
        glm::mat4 M(1.f);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) M[c][r] = (float)m9[3 * r + c];   // MuJoCo is row-major, glm column-major
        M[3] = glm::vec4((float)pos[0], (float)pos[1], (float)pos[2], 1.f);
        return M;
    };
    for (int b = 0; b < _m->nbody; ++b) _snap.bodyPose[b] = mat(_d->xpos + 3 * b, _d->xmat + 9 * b);
    for (int g = 0; g < _m->ngeom; ++g) _snap.geomPose[g] = mat(_d->geom_xpos + 3 * g, _d->geom_xmat + 9 * g);
    for (int l = 0; l < kLegs; ++l) _snap.footSim[l] = Rt * (col3(_d->xpos + 3 * mj_name2id(_m, mjOBJ_BODY, _legs[l].toolBody.c_str())) - s.pos);
    ++_snap.steps;
}

// The exchange with the app, on the main thread, under a short lock.
void QuadrupedClient::update() {
    std::lock_guard<std::mutex> lk(_mtx);
    bool all = true;
    for (int l = 0; l < kLegs; ++l) {
        if (!_state[l]) return;
        _inTarget[l] = _state[l]->targetJoints;
        all = all && _state[l]->hasTarget;
    }
    _inHasTarget = all;
    if (!_snap.sense.valid) return;
    for (int l = 0; l < kLegs; ++l) {
        RobotState& st = *_state[l];
        st.currentJoints = JointVec(kLegJoints);
        st.commandedJoints = JointVec(kLegJoints);
        for (int j = 0; j < kLegJoints; ++j) {
            st.currentJoints[j] = _snap.sense.q[l * kLegJoints + j] / kRad;
            st.commandedJoints[j] = _snap.cmdDeg[l * kLegJoints + j];
        }
        st.egmConnected = _connected.load();    // "connected", whatever the transport
        st.timestamp = _snap.sense.simTime;
        st.packetsRx = _snap.steps;
        st.packetsTx = _snap.streamed;
    }
}

QuadrupedClient::Sense QuadrupedClient::sense() const {
    std::lock_guard<std::mutex> lk(_mtx);
    return _snap.sense;
}

void QuadrupedClient::bodyPoses(std::vector<glm::mat4>& out) const {
    std::lock_guard<std::mutex> lk(_mtx);
    out = _snap.bodyPose;
}

void QuadrupedClient::geomPoses(std::vector<glm::mat4>& out) const {
    std::lock_guard<std::mutex> lk(_mtx);
    out = _snap.geomPose;
}

bool QuadrupedClient::runActuator(int n, std::vector<float>& x, std::vector<float>& h, std::vector<float>& c, std::vector<float>& torque) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<int64_t> sx{n, 1, 2}, sh{2, n, 8};
    std::vector<Ort::Value> in(3);
    for (int i = 0; i < 3; ++i) {
        switch (_actInIdx[i]) {
            case 0: in[i] = _actNet.getInputCpuTensor(x, sx); break;
            case 1: in[i] = _actNet.getInputCpuTensor(h, sh); break;
            case 2: in[i] = _actNet.getInputCpuTensor(c, sh); break;
            default: return false;
        }
    }
    try {
        std::vector<Ort::Value> out = _actNet.getSession().Run(Ort::RunOptions{nullptr}, _actIn.data(), in.data(), 3, _actOut.data(), 3);
        for (int i = 0; i < 3; ++i) {
            const float* p = out[i].GetTensorData<float>();
            switch (_actOutIdx[i]) {
                case 0: torque.assign(p, p + n); break;
                case 1: std::copy(p, p + 2 * n * 8, h.begin()); break;
                case 2: std::copy(p, p + 2 * n * 8, c.begin()); break;
                default: return false;
            }
        }
    } catch (const Ort::Exception& e) {
        ofLogError("QuadrupedClient") << "actuator network: " << e.what();
        return false;
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    _actuatorMicros = 0.98 * _actuatorMicros.load() + 0.02 * us;
    return true;
}

// Where ofxRobot's kinematics put a leg's foot for joint angles `q` (degrees), in the trunk frame.
glm::dmat4 QuadrupedClient::footFk(int leg, const JointVec& q) const {
    return _legs[leg].ik.linkFrames(q).back() * _legs[leg].tool;
}

glm::dvec3 QuadrupedClient::toolFk(int leg) const {
    const Sense s = sense();
    JointVec q(kLegJoints);
    for (int j = 0; j < kLegJoints; ++j) q[j] = s.q[leg * kLegJoints + j] / kRad;
    return glm::dvec3(footFk(leg, q)[3]);
}

glm::dvec3 QuadrupedClient::toolSim(int leg) const {
    std::lock_guard<std::mutex> lk(_mtx);
    return _snap.footSim[leg];
}

double QuadrupedClient::maxFkErrorMm() const {
    double worst = 0.0;
    for (int l = 0; l < kLegs; ++l) worst = std::max(worst, glm::length(toolFk(l) - toolSim(l)) * 1000.0);
    return worst;
}
