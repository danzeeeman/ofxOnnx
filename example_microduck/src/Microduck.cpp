#include "Microduck.h"
#include <chrono>

namespace {
glm::dvec3 col3(const mjtNum* v) { return glm::dvec3(v[0], v[1], v[2]); }
}

Microduck::~Microduck() {
    stopThread();
    if (_d) mj_deleteData(_d);
    if (_m) mj_deleteModel(_m);
}

void Microduck::startThread() {
    if (_thread.joinable() || !_m) return;
    _run = true;
    _thread = std::thread([this] {
        using clock = std::chrono::steady_clock;
        auto last = clock::now();
        while (_run) {
            const auto now = clock::now();
            const double dt = std::chrono::duration<double>(now - last).count();
            last = now;
            {
                std::lock_guard<std::mutex> lock(mutex);
                update(dt);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
}

void Microduck::stopThread() {
    _run = false;
    if (_thread.joinable()) _thread.join();
}

bool Microduck::setup(const std::string& dir) {
    const std::string base = ofToDataPath(dir, true) + "/";
    if (!ofFile::doesFileExist(base + "microduck.xml") || !ofFile::doesFileExist(base + "microduck.json")) {
        ofLogError("Microduck") << "no model in " << base << ": run tools/make_microduck.py (see the README)";
        return false;
    }
    const ofJson meta = ofLoadJson(dir + "/microduck.json");

    // 1. Physics: the model, with the servos and the floor in it.
    char error[1000] = "";
    _m = mj_loadXML((base + "microduck.xml").c_str(), nullptr, error, sizeof(error));
    if (!_m) { ofLogError("Microduck") << "MuJoCo could not load the model: " << error; return false; }
    _d = mj_makeData(_m);
    _trunk = mj_name2id(_m, mjOBJ_BODY, meta["trunk_body"].get<std::string>().c_str());
    if (_trunk < 0) { ofLogError("Microduck") << "no trunk body"; return false; }
    _simDt = meta["sim_dt"].get<double>();
    _decimation = meta["decimation"].get<int>();
    _initHeight = meta["init_height"].get<float>();
    if (std::fabs(_m->opt.timestep - _simDt) > 1e-9)
        ofLogWarning("Microduck") << "the model's time step " << _m->opt.timestep << " is not the policy's " << _simDt;

    // 2. The policy's two orderings: the observation's joints, and the action's joints (legs, then head).
    _jointOrder = meta["joint_order"].get<std::vector<std::string>>();
    _actionOrder = meta["action_order"].get<std::vector<std::string>>();
    const std::vector<float> scale = meta["action_scale"].get<std::vector<float>>();
    for (const std::string& n : _jointOrder) {
        const int j = mj_name2id(_m, mjOBJ_JOINT, n.c_str());
        if (j < 0) { ofLogError("Microduck") << "no joint " << n; return false; }
        _obsQpos.push_back(_m->jnt_qposadr[j]);
        _obsDof.push_back(_m->jnt_dofadr[j]);
        _defaultObs.push_back(meta["default_pos"][n].get<float>());
    }
    for (size_t i = 0; i < _actionOrder.size(); ++i) {
        const int a = mj_name2id(_m, mjOBJ_ACTUATOR, _actionOrder[i].c_str());
        if (a < 0) { ofLogError("Microduck") << "no actuator " << _actionOrder[i]; return false; }
        _actuator.push_back(a);
        _defaultAct.push_back(meta["default_pos"][_actionOrder[i]].get<float>());
        _scale.push_back(scale[i]);
    }
    const auto& r = meta["command_range"];
    _rangeX = {r["vx"][0].get<float>(), r["vx"][1].get<float>()};
    _rangeY = {r["vy"][0].get<float>(), r["vy"][1].get<float>()};
    _rangeZ = {r["wz"][0].get<float>(), r["wz"][1].get<float>()};

    // 2b. The sensors: the IMUs' accelerometers and gyros, and the time-of-flight grid.
    auto sensorAdr = [&](const char* key) {
        const int id = mj_name2id(_m, mjOBJ_SENSOR, meta["sensors"][key].get<std::string>().c_str());
        return id < 0 ? -1 : _m->sensor_adr[id];
    };
    _accBody = sensorAdr("body_acc"); _gyroBody = sensorAdr("body_gyro"); _accHead = sensorAdr("head_acc"); _gyroHead = sensorAdr("head_gyro");
    _headBody = mj_name2id(_m, mjOBJ_BODY, meta["head_body"].get<std::string>().c_str());
    if (_accBody < 0 || _gyroBody < 0 || _accHead < 0 || _gyroHead < 0 || _headBody < 0) { ofLogError("Microduck") << "the model lacks the IMU sensors"; return false; }
    _skill = meta["skill"].get<int>();
    _numSkills = meta["num_skills"].get<int>();
    {
        const auto& t = meta["tof"];
        _tof.pos = glm::dvec3(t["pos"][0].get<double>(), t["pos"][1].get<double>(), t["pos"][2].get<double>());
        _tof.pitchUp = t["pitch_up_deg"].get<double>(); _tof.fov = t["fov_deg"].get<double>(); _tof.zones = t["zones"].get<int>();
        _tof.maxDistance = t["max_distance"].get<double>(); _tof.minDistance = t["min_distance"].get<double>();
        // Isaac Lab's lidar pattern: `zones` x `zones` angles over the field of view (an edge zone's centre is fov/2 * (zones-1)/zones
        // from the axis), the rays ordered vertical-major, tilted up by the mount pitch (a rotation about y by -pitch).
        const double rad = 3.14159265358979 / 180.0, edge = _tof.fov / 2.0 * (_tof.zones - 1) / _tof.zones;
        const double tilt = -_tof.pitchUp * rad, ct = std::cos(tilt), st = std::sin(tilt);
        for (int v = 0; v < _tof.zones; ++v)
            for (int hh = 0; hh < _tof.zones; ++hh) {
                const double va = (-edge + 2.0 * edge * v / (_tof.zones - 1)) * rad, ha = (-edge + 2.0 * edge * hh / (_tof.zones - 1)) * rad;
                const glm::dvec3 d(std::cos(va) * std::cos(ha), std::cos(va) * std::sin(ha), std::sin(va));
                _tofDirs.push_back(glm::dvec3(ct * d.x + st * d.z, d.y, -st * d.x + ct * d.z));   // Ry(tilt)
            }
    }

    // 3. The brain.
    ofxOnnx::Settings settings;
    settings.modelPath = dir + "/walker.onnx";
    settings.numIntraOpThreads = 1;           // a tiny network: threads would only add cost
    settings.logLevel = OF_LOG_WARNING;
    settings.bPrintModelInfo = false;
    if (!_onnx.load(settings)) return false;
    _obs.assign(meta["obs_dim"].get<int>(), 0.f);
    _last.assign(14, 0.f);

    // 4. The robot as ofxRobot sees it: one serial chain per limb, from a JSON descriptor, and its forward kinematics.
    for (auto it = meta["limbs"].begin(); it != meta["limbs"].end(); ++it) {
        _limbs.emplace_back();
        Limb& l = _limbs.back();
        l.name = it.key();
        if (!l.kin.load(ofToDataPath(dir + "/" + it.value().get<std::string>(), true))) return false;
        std::vector<ArmIk::JointGeom> chain(l.kin.numJoints());
        for (int i = 0; i < l.kin.numJoints(); ++i) {
            const auto& j = l.kin.joint(i);
            chain[i].origin = glm::dvec3(j.origin);
            chain[i].rot0 = glm::dquat(j.rot0);
            chain[i].rot1 = glm::dquat(j.rot1);
            l.kin.limits(i, chain[i].lo, chain[i].hi);
            const auto name = std::find(_actionOrder.begin(), _actionOrder.end(), j.name);
            if (name == _actionOrder.end()) { ofLogError("Microduck") << l.name << ": unknown joint " << j.name; return false; }
            l.actionIdx.push_back((int)(name - _actionOrder.begin()));
            l.qposAdr.push_back(_m->jnt_qposadr[mj_name2id(_m, mjOBJ_JOINT, j.name.c_str())]);
        }
        std::string why;
        if (!ArmIk::paramsFromChainFkOnly(chain, l.ik.params, why)) { ofLogError("Microduck") << l.name << ": " << why; return false; }

        // The tool frame (the foot, the mouth tip): a fixed frame after the last joint.
        const ofJson desc = ofLoadJson(dir + "/" + it.value().get<std::string>());
        const auto& o = desc["tool"]["origin"];
        const auto& rpy = desc["tool"]["rpy"];
        l.tool = glm::translate(glm::dmat4(1.0), glm::dvec3(o[0].get<double>(), o[1].get<double>(), o[2].get<double>())) *
                 glm::mat4_cast(glm::dquat(glm::dvec3(rpy[0].get<double>(), rpy[1].get<double>(), rpy[2].get<double>())));
        l.toolBody = desc["tool"]["link"].get<std::string>();
        if (mj_name2id(_m, mjOBJ_BODY, l.toolBody.c_str()) < 0) { ofLogError("Microduck") << "no body " << l.toolBody; return false; }

        // The targets come from a neural network at 50 Hz and jump by tens of degrees between steps by design, so
        // the gate is set to catch only what is wrong: NaNs, and jumps no servo could follow.
        l.gate.params.maxJumpDegPerSec = 6000.f;
        l.q = JointVec(l.kin.numJoints());
        l.target = JointVec(l.kin.numJoints());
    }
    reset();
    ofLogNotice("Microduck") << "ready: " << _m->nbody << " bodies, " << _limbs.size() << " limbs (ofxRobot), policy " << settings.modelPath;
    return true;
}

void Microduck::reset() {
    mj_resetData(_m, _d);
    _d->qpos[0] = 0.0; _d->qpos[1] = 0.0; _d->qpos[2] = _initHeight;
    _d->qpos[3] = 1.0; _d->qpos[4] = 0.0; _d->qpos[5] = 0.0; _d->qpos[6] = 0.0;
    for (size_t i = 0; i < _obsQpos.size(); ++i) _d->qpos[_obsQpos[i]] = _defaultObs[i];
    for (size_t i = 0; i < _actuator.size(); ++i) _d->ctrl[_actuator[i]] = _defaultAct[i];
    mj_forward(_m, _d);
    _headingTarget = heading();
    {   // eye level: the head camera (jaw_soft body, offset in its frame) in the default pose, above the trunk
        const int head = mj_name2id(_m, mjOBJ_BODY, "jaw_soft");
        if (head >= 0) {
            const double* R = _d->xmat + 9 * head;
            const double off[3] = {0.0155, 0.0, -0.0733};
            const double eyeZ = _d->xpos[3 * head + 2] + R[6] * off[0] + R[7] * off[1] + R[8] * off[2];
            _eyeHeight = eyeZ - _d->xpos[3 * _trunk + 2];
        }
    }
    std::fill(_last.begin(), _last.end(), 0.f);
    _accum = 0.0;
    clearFault();
    for (Limb& l : _limbs) {
        l.gate.clearFault();
        for (int i = 0; i < l.q.size(); ++i) l.q[i] = l.target[i] = _d->qpos[l.qposAdr[i]] * 180.0 / 3.14159265358979;
    }
}

void Microduck::clearFault() {
    _faulted = false;
    _faultReason.clear();
    for (Limb& l : _limbs) l.gate.clearFault();
}

void Microduck::setCommand(float vx, float vy, float wz) {
    _cmd = glm::vec3(glm::clamp(vx, _rangeX.x, _rangeX.y), glm::clamp(vy, _rangeY.x, _rangeY.y), glm::clamp(wz, _rangeZ.x, _rangeZ.y));
}

void Microduck::update(double wallSeconds) {
    if (paused || !_m) return;
    const double period = _decimation * _simDt;
    _accum += std::min(wallSeconds, 0.1);   // after a long stall, do not try to catch up all of it
    while (_accum >= period) {
        policyStep();
        _accum -= period;
    }
}

glm::dvec3 Microduck::trunkPos() const { return col3(_d->xpos + 3 * _trunk); }

glm::dmat3 Microduck::trunkRot() const {
    const mjtNum* x = _d->xmat + 9 * _trunk;    // row-major, body to world
    glm::dmat3 R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R[c][r] = x[3 * r + c];   // glm is column-major
    return R;
}

glm::dvec3 Microduck::velocityBody() const { return glm::transpose(trunkRot()) * col3(_d->qvel); }
double Microduck::uprightness() const { return _d->xmat[9 * _trunk + 8]; }

// The policy's 136 inputs, in the order it was trained with.
double Microduck::heading() const {
    const glm::dmat3 R = trunkRot();
    return std::atan2(R[0][1], R[0][0]);
}

void Microduck::buildObservation() {
    const glm::dmat3 R = trunkRot();
    const glm::dmat3 Rt = glm::transpose(R);
    const double* sd = _d->sensordata;
    auto s3 = [&](int adr) { return glm::dvec3(sd[adr], sd[adr + 1], sd[adr + 2]); };
    size_t k = 0;
    auto put = [&](const glm::dvec3& v, double scale = 1.0) { for (int i = 0; i < 3; ++i) _obs[k++] = (float)(v[i] * scale); };
    put(s3(_gyroBody));                                                 // base_ang_vel: the body IMU's gyro
    put(Rt * glm::dvec3(0.0, 0.0, -1.0));                               // projected_gravity, in the trunk frame
    put(s3(_accBody), 0.1);                                             // base_lin_acc: the accelerometer (proper acceleration), x 0.1
    for (size_t i = 0; i < _obsQpos.size(); ++i) _obs[k++] = (float)_d->qpos[_obsQpos[i]] - _defaultObs[i];   // joint_pos_rel
    for (size_t i = 0; i < _obsDof.size(); ++i) _obs[k++] = (float)_d->qvel[_obsDof[i]];                      // joint_vel_rel
    for (size_t i = 0; i < _last.size(); ++i) _obs[k++] = _last[i];                                           // last_action
    put(s3(_gyroHead));                                                 // head_ang_vel
    put(s3(_accHead), 0.1);                                             // head_lin_acc
    {   // head_gravity: the direction of gravity in the head body's frame
        const double* Rh = _d->xmat + 9 * _headBody;
        put(glm::dvec3(-Rh[6], -Rh[7], -Rh[8]));
    }
    {   // tof: distances of the 8x8 rays out of the trunk, over the range; nothing seen (or only the robot's own body) is 1
        const glm::dvec3 origin = trunkPos() + R * _tof.pos;
        const mjtByte collisionGroup[mjNGROUP] = {1, 0, 0, 0, 0, 0};    // the collision shapes: the floor and the robot's, not its visual meshes
        for (const glm::dvec3& dl : _tofDirs) {
            const glm::dvec3 dir = R * dl;
            int geom = -1;
            double dist = mj_ray(_m, _d, &origin.x, &dir.x, collisionGroup, 1, _trunk, &geom, nullptr);
            if (dist < 0.0 || dist < _tof.minDistance) dist = _tof.maxDistance;
            _obs[k++] = (float)glm::clamp(dist / _tof.maxDistance, 0.0, 1.0);
        }
    }
    for (int i = 0; i < _numSkills; ++i) _obs[k++] = i == _skill ? 1.f : 0.f;     // the skill command: walk
    for (int i = 0; i < 3; ++i) _obs[k++] = _applied[i];                          // velocity_command: vx, vy, wz
    for (int i = 0; i < 3; ++i) _obs[k++] = 0.f;                                  // target: zeros, the walk skill has none
}

glm::dvec3 Microduck::lookTarget() const {
    const double yaw = heading();
    // it drifts left and right of dead ahead, a few centimetres either way, so the head has something to follow
    const double side = lookSwayMetres * std::sin(2.0 * 3.14159265358979 * simTime() / lookSwaySeconds);
    return trunkPos() + glm::dvec3(lookDistance * std::cos(yaw) - side * std::sin(yaw),
                                   lookDistance * std::sin(yaw) + side * std::cos(yaw), _eyeHeight);
}

void Microduck::policyStep() {
    std::vector<float> targets = _defaultAct;               // rad; the default pose is what a faulted robot holds

    // The turn rate the policy is given: yours while you turn, otherwise whatever holds the heading.
    {
        const double yaw = heading();
        _applied = _cmd;
        if (std::fabs(_cmd.z) > 1e-3f) {
            _headingTarget = yaw;
        } else {
            double err = _headingTarget - yaw;
            err = std::atan2(std::sin(err), std::cos(err));             // wrapped to (-pi, pi]
            _applied.z = glm::clamp(kHeadingStiffness * (float)err, _rangeZ.x, _rangeZ.y);
        }
    }

    if (!_faulted) {
        buildObservation();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = _onnx.runFloat(_obs, _act);
        _policyMicros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (!ok || _act.size() != 14) {
            _faulted = true; _faultReason = "the policy did not return 14 actions";
        } else {
            _last = _act;
            for (size_t i = 0; i < targets.size(); ++i) targets[i] = _defaultAct[i] + _scale[i] * _act[i];
        }
    }

    // ofxRobot's gate, per limb, on what is about to be sent (degrees, as everywhere in ofxRobot).
    if (!_faulted) {
        for (Limb& l : _limbs) {
            for (int i = 0; i < l.target.size(); ++i) l.target[i] = targets[l.actionIdx[i]] * 180.0 / 3.14159265358979;
            if (!l.gate.observe(l.target, (float)(_decimation * _simDt))) {
                _faulted = true;
                _faultReason = l.name + ": " + l.gate.faultReason();
                break;
            }
        }
        if (_faulted) targets = _defaultAct;
    }
    for (size_t i = 0; i < _actuator.size(); ++i) _d->ctrl[_actuator[i]] = targets[i];

    for (int k = 0; k < _decimation; ++k) mj_step(_m, _d);
    // mj_step integrates after it computes the bodies' poses, so xpos/xmat lag qpos by a step: refresh them, so that what
    // is read from them (the drawing, the forward-kinematics check) belongs to the same state as the joint angles.
    mj_kinematics(_m, _d);

    if (!std::isfinite(_d->qpos[2])) { reset(); return; }   // the physics diverged: start over
    for (Limb& l : _limbs)
        for (int i = 0; i < l.q.size(); ++i) l.q[i] = _d->qpos[l.qposAdr[i]] * 180.0 / 3.14159265358979;
}

void Microduck::push(float vx, float vy) {
    const glm::dvec3 w = trunkRot() * glm::dvec3(vx, vy, 0.0);
    _d->qvel[0] += w.x;
    _d->qvel[1] += w.y;
}

// Where ofxRobot's kinematics put the tool, in the trunk frame.
glm::dvec3 Microduck::toolFk(const Limb& l) const {
    JointVec q(l.kin.numJoints());
    for (int i = 0; i < q.size(); ++i) q[i] = _d->qpos[l.qposAdr[i]] * 180.0 / 3.14159265358979;
    return glm::dvec3((l.ik.linkFrames(q).back() * l.tool)[3]);
}

// Where the simulation has it, in the same frame.
glm::dvec3 Microduck::toolSim(const Limb& l) const {
    const int b = mj_name2id(_m, mjOBJ_BODY, l.toolBody.c_str());
    return glm::transpose(trunkRot()) * (col3(_d->xpos + 3 * b) - trunkPos());
}

double Microduck::maxFkErrorMm() const {
    double worst = 0.0;
    for (const Limb& l : _limbs) worst = std::max(worst, glm::length(toolFk(l) - toolSim(l)) * 1000.0);
    return worst;
}
