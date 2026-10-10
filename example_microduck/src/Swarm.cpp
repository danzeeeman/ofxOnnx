#include "Swarm.h"
#include <chrono>

namespace {
const double kPi = 3.14159265358979;
glm::dvec3 col3(const double* v) { return glm::dvec3(v[0], v[1], v[2]); }
double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }   // to (-pi, pi]
}

Swarm::~Swarm() {
    stopThread();
    if (_d) mj_deleteData(_d);
    if (_m) mj_deleteModel(_m);
}

bool Swarm::setup(const std::string& dir) {
    const std::string base = ofToDataPath(dir, true) + "/";
    if (!ofFile::doesFileExist(base + "swarm.xml") || !ofFile::doesFileExist(base + "navigator.onnx")) {
        ofLogError("Swarm") << "no swarm.xml / navigator.onnx in " << base << " (tools/make_swarm.py makes them)";
        return false;
    }
    const ofJson meta = ofLoadJson(dir + "/microduck.json");

    char error[1000] = "";
    _m = mj_loadXML((base + "swarm.xml").c_str(), nullptr, error, sizeof(error));
    if (!_m) { ofLogError("Swarm") << "MuJoCo could not load the world: " << error; return false; }
    _d = mj_makeData(_m);
    _simDt = meta["sim_dt"].get<double>();
    _decimation = meta["decimation"].get<int>();
    _initHeight = meta["init_height"].get<double>();
    _numSkills = meta["num_skills"].get<int>();
    _skillIndex = meta["policies"]["walk"]["skill"].get<int>();      // the navigation policy is the walk skill, trained among obstacles

    const auto jointOrder = meta["joint_order"].get<std::vector<std::string>>();
    const auto actionOrder = meta["action_order"].get<std::vector<std::string>>();
    const auto scale = meta["action_scale"].get<std::vector<float>>();
    for (size_t i = 0; i < actionOrder.size(); ++i) {
        _defaultAct.push_back(meta["default_pos"][actionOrder[i]].get<float>());
        _scale.push_back(scale[i]);
    }
    for (const std::string& n : jointOrder) _defaultObs.push_back(meta["default_pos"][n].get<float>());
    {
        const auto& t = meta["tof"];
        _tof.pos = glm::dvec3(t["pos"][0].get<double>(), t["pos"][1].get<double>(), t["pos"][2].get<double>());
        _tof.pitchUp = t["pitch_up_deg"].get<double>(); _tof.fov = t["fov_deg"].get<double>(); _tof.zones = t["zones"].get<int>();
        _tof.maxDistance = t["max_distance"].get<double>(); _tof.minDistance = t["min_distance"].get<double>();
        const double rad = kPi / 180.0, edge = _tof.fov / 2.0 * (_tof.zones - 1) / _tof.zones;
        const double tilt = -_tof.pitchUp * rad, ct = std::cos(tilt), st = std::sin(tilt);
        for (int v = 0; v < _tof.zones; ++v)
            for (int h = 0; h < _tof.zones; ++h) {
                const double va = (-edge + 2.0 * edge * v / (_tof.zones - 1)) * rad, ha = (-edge + 2.0 * edge * h / (_tof.zones - 1)) * rad;
                const glm::dvec3 d(std::cos(va) * std::cos(ha), std::cos(va) * std::sin(ha), std::sin(va));
                _tofDirs.push_back(glm::dvec3(ct * d.x + st * d.z, d.y, -st * d.x + ct * d.z));
            }
    }

    // The ducks: d0_, d1_, ... in the world, each with its own joints, servos and sensors.
    for (int k = 0;; ++k) {
        const std::string p = "d" + std::to_string(k) + "_";
        const int trunk = mj_name2id(_m, mjOBJ_BODY, (p + meta["trunk_body"].get<std::string>()).c_str());
        if (trunk < 0) break;
        _ducks.emplace_back();
        Duck& d = _ducks.back();
        d.trunk = trunk;
        d.head = mj_name2id(_m, mjOBJ_BODY, (p + meta["head_body"].get<std::string>()).c_str());
        d.freeQpos = _m->jnt_qposadr[_m->body_jntadr[trunk]];
        for (const std::string& n : jointOrder) {
            const int j = mj_name2id(_m, mjOBJ_JOINT, (p + n).c_str());
            if (j < 0) { ofLogError("Swarm") << "no joint " << p + n; return false; }
            d.obsQpos.push_back(_m->jnt_qposadr[j]);
            d.obsDof.push_back(_m->jnt_dofadr[j]);
        }
        for (const std::string& n : actionOrder) {
            const int a = mj_name2id(_m, mjOBJ_ACTUATOR, (p + n).c_str());
            if (a < 0) { ofLogError("Swarm") << "no actuator " << p + n; return false; }
            d.actuator.push_back(a);
        }
        auto sensorAdr = [&](const char* key) {
            const int id = mj_name2id(_m, mjOBJ_SENSOR, (p + meta["sensors"][key].get<std::string>()).c_str());
            return id < 0 ? -1 : _m->sensor_adr[id];
        };
        d.accBody = sensorAdr("body_acc"); d.gyroBody = sensorAdr("body_gyro"); d.accHead = sensorAdr("head_acc"); d.gyroHead = sensorAdr("head_gyro");
        if (d.head < 0 || d.accBody < 0 || d.gyroBody < 0 || d.accHead < 0 || d.gyroHead < 0) { ofLogError("Swarm") << "duck " << k << " lacks its IMUs"; return false; }
        d.obs.assign(meta["obs_dim"].get<int>(), 0.f);
        d.act.assign(14, 0.f);
        d.last.assign(14, 0.f);
        // ofxRobot's gate per limb: the left leg, the right leg and the head
        d.limbs.resize(3);
        for (size_t i = 0; i < actionOrder.size(); ++i) {
            const std::string& n = actionOrder[i];
            const int limb = n.rfind("left_", 0) == 0 ? 0 : n.rfind("right_", 0) == 0 ? 1 : 2;
            d.limbs[limb].actionIdx.push_back((int)i);
        }
        for (auto& l : d.limbs) {
            l.gate.params.maxJumpDegPerSec = 20000.f;       // a neural network's targets jump by tens of degrees by design
            l.target = JointVec((int)l.actionIdx.size());
        }
    }
    if (_ducks.empty()) { ofLogError("Swarm") << "no ducks in swarm.xml"; return false; }

    _bodyDuck.assign(_m->nbody, -1);
    for (int b = 0; b < _m->nbody; ++b)
        for (size_t k = 0; k < _ducks.size(); ++k)
            if (_m->body_rootid[b] == _ducks[k].trunk) _bodyDuck[b] = (int)k;
    _geomDuck.assign(_m->ngeom, -1);
    for (int g = 0; g < _m->ngeom; ++g) {
        _geomDuck[g] = _bodyDuck[_m->geom_bodyid[g]];
        const char* name = mj_id2name(_m, mjOBJ_GEOM, g);
        if (name && std::string(name).rfind("wall", 0) == 0) _geomDuck[g] = -2;
        if (name && std::string(name).rfind("obstacle", 0) == 0) {
            _geomDuck[g] = -3;
            const double* s = _m->geom_size + 3 * g;
            _obstacles.emplace_back(_m->geom_pos[3 * g], _m->geom_pos[3 * g + 1], _m->geom_type[g] == mjGEOM_BOX ? std::hypot(s[0], s[1]) : s[0]);
        }
    }
    {   // the arena's size, from its walls
        const int w = mj_name2id(_m, mjOBJ_GEOM, "wall0");
        if (w >= 0) _half = _m->geom_pos[3 * w + 1] - 0.05;
    }

    ofxOnnx::Settings settings;
    settings.modelPath = dir + "/navigator.onnx";
    settings.numIntraOpThreads = 1;
    settings.logLevel = OF_LOG_WARNING;
    settings.bPrintModelInfo = false;
    if (!_onnx.load(settings)) return false;

    reset();
    ofLogNotice("Swarm") << _ducks.size() << " ducks in a " << 2 * _half << " m square arena, policy " << settings.modelPath;
    return true;
}

bool Swarm::clearOfObstacles(const glm::dvec2& p, double margin) const {
    for (const glm::dvec3& o : _obstacles) if (glm::length(p - glm::dvec2(o)) < o.z + margin) return false;
    return true;
}

void Swarm::reset() {
    mj_resetData(_m, _d);
    // scatter the ducks over a disc inside the walls, at least 0.5 m apart, each facing a random way
    const double radius = _half - 0.4;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<glm::dvec2> placed;
    for (Duck& d : _ducks) {
        glm::dvec2 p(0.0);
        bool found = false;
        for (double apart : {0.5, 0.4, 0.3}) {                      // as far apart as the crowd allows
            for (int tries = 0; tries < 400 && !found; ++tries) {
                const double r = radius * std::sqrt(uni(_rng)), a = 2.0 * kPi * uni(_rng);
                p = glm::dvec2(r * std::cos(a), r * std::sin(a));
                bool clear = clearOfObstacles(p, 0.35);
                for (const glm::dvec2& q : placed) if (glm::length(p - q) < apart) { clear = false; break; }
                found = clear;
            }
            if (found) break;
        }
        placed.push_back(p);
        const double yaw = (2.0 * uni(_rng) - 1.0) * kPi;
        double* q = _d->qpos + d.freeQpos;
        q[0] = p.x; q[1] = p.y; q[2] = _initHeight;
        q[3] = std::cos(yaw / 2.0); q[4] = 0.0; q[5] = 0.0; q[6] = std::sin(yaw / 2.0);
        for (size_t i = 0; i < d.obsQpos.size(); ++i) _d->qpos[d.obsQpos[i]] = _defaultObs[i];
        for (size_t i = 0; i < d.actuator.size(); ++i) _d->ctrl[d.actuator[i]] = _defaultAct[i];
        std::fill(d.last.begin(), d.last.end(), 0.f);
        d.vx = (float)(0.2 + 0.2 * uni(_rng));                  // each walks at its own pace, 0.2-0.4 m/s
        d.headingTarget = yaw;
        d.nextHeading = 5.0 + 10.0 * uni(_rng);
        d.faulted = false;
        for (auto& l : d.limbs) l.gate.clearFault();
    }
    mj_forward(_m, _d);
    _stats = Stats();
    _tObserve = _tInfer = _tStep = 0; _periods = 0;
    _touched.assign(_ducks.size(), std::vector<char>(_ducks.size(), 0));
}

glm::dvec3 Swarm::pos(int k) const { return col3(_d->xpos + 3 * _ducks[k].trunk); }
double Swarm::heading(int k) const { const double* R = _d->xmat + 9 * _ducks[k].trunk; return std::atan2(R[3], R[0]); }
double Swarm::uprightness(int k) const { return _d->xmat[9 * _ducks[k].trunk + 8]; }

void Swarm::buildObservation(Duck& d) {
    const double* Rm = _d->xmat + 9 * d.trunk;                       // body to world, row-major
    auto toBody = [&](const glm::dvec3& w) { return glm::dvec3(Rm[0] * w.x + Rm[3] * w.y + Rm[6] * w.z, Rm[1] * w.x + Rm[4] * w.y + Rm[7] * w.z, Rm[2] * w.x + Rm[5] * w.y + Rm[8] * w.z); };
    const double* sd = _d->sensordata;
    auto s3 = [&](int adr) { return glm::dvec3(sd[adr], sd[adr + 1], sd[adr + 2]); };
    size_t k = 0;
    auto put = [&](const glm::dvec3& v, double scale = 1.0) { for (int i = 0; i < 3; ++i) d.obs[k++] = (float)(v[i] * scale); };
    put(s3(d.gyroBody));                                             // base_ang_vel: the body IMU's gyro
    put(toBody(glm::dvec3(0.0, 0.0, -1.0)));                         // projected_gravity
    put(s3(d.accBody), 0.1);                                         // base_lin_acc: the accelerometer x 0.1
    for (size_t i = 0; i < d.obsQpos.size(); ++i) d.obs[k++] = (float)_d->qpos[d.obsQpos[i]] - _defaultObs[i];
    for (size_t i = 0; i < d.obsDof.size(); ++i) d.obs[k++] = (float)_d->qvel[d.obsDof[i]];
    for (size_t i = 0; i < d.last.size(); ++i) d.obs[k++] = d.last[i];
    put(s3(d.gyroHead));
    put(s3(d.accHead), 0.1);
    {
        const double* Rh = _d->xmat + 9 * d.head;
        put(glm::dvec3(-Rh[6], -Rh[7], -Rh[8]));                     // head_gravity
    }
    {   // tof: 8x8 rays out of the chest; the walls and the other ducks are what they hit besides the floor
        const glm::dvec3 origin = col3(_d->xpos + 3 * d.trunk) + glm::dvec3(Rm[0] * _tof.pos.x + Rm[1] * _tof.pos.y + Rm[2] * _tof.pos.z,
                                                                          Rm[3] * _tof.pos.x + Rm[4] * _tof.pos.y + Rm[5] * _tof.pos.z,
                                                                          Rm[6] * _tof.pos.x + Rm[7] * _tof.pos.y + Rm[8] * _tof.pos.z);
        const mjtByte collisionGroup[mjNGROUP] = {1, 0, 0, 0, 0, 0};
        d.raysOnDucks = d.raysOnWalls = 0;
        for (const glm::dvec3& dl : _tofDirs) {
            const glm::dvec3 dir(Rm[0] * dl.x + Rm[1] * dl.y + Rm[2] * dl.z, Rm[3] * dl.x + Rm[4] * dl.y + Rm[5] * dl.z, Rm[6] * dl.x + Rm[7] * dl.y + Rm[8] * dl.z);
            int geom = -1;
            double dist = mj_ray(_m, _d, &origin.x, &dir.x, collisionGroup, 1, d.trunk, &geom, nullptr);   // reads the world, writes nothing: safe on many threads
            if (dist < 0.0 || dist < _tof.minDistance) dist = _tof.maxDistance;
            else if (geom >= 0) {
                const int owner = _geomDuck[geom];
                if (owner >= 0 && owner != (int)(&d - &_ducks[0])) ++d.raysOnDucks;
                else if (owner == -2 || owner == -3) ++d.raysOnWalls;
            }
            d.obs[k++] = (float)glm::clamp(dist / _tof.maxDistance, 0.0, 1.0);
        }
    }
    for (int i = 0; i < _numSkills; ++i) d.obs[k++] = i == _skillIndex ? 1.f : 0.f;
    for (int i = 0; i < 3; ++i) d.obs[k++] = d.applied[i];           // velocity_command: forward, 0, the turn the heading controller asks for
    for (int i = 0; i < 3; ++i) d.obs[k++] = 0.f;                    // target: none
}

// Every duck's command and observation. A duck walks forwards at its own pace; its heading target wanders, and the turn
// rate is the heading controller's (stiffness 0.5, as in training). Steering clear of walls and ducks is the policy's.
void Swarm::observe() {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (Duck& d : _ducks) {
        if (_d->time >= d.nextHeading) {
            d.headingTarget = (2.0 * uni(_rng) - 1.0) * kPi;
            d.nextHeading = _d->time + 8.0 + 8.0 * uni(_rng);
        }
        const double err = wrap(d.headingTarget - heading((int)(&d - &_ducks[0])));
        d.applied = glm::vec3(d.vx, 0.f, glm::clamp(kHeadingStiffness * (float)err, -1.f, 1.f));
    }
    // The observations, the time-of-flight rays above all, are the bulk of a period's work and each duck's is independent, so
    // the ducks are shared out over a few threads (they only read the world).
    const int n = (int)_ducks.size();
    const int workers = std::max(1, std::min({(int)std::thread::hardware_concurrency(), 8, n}));
    auto work = [&](int w) { for (int k = w; k < n; k += workers) if (!_ducks[k].faulted) buildObservation(_ducks[k]); };
    std::vector<std::thread> pool;
    for (int w = 1; w < workers; ++w) pool.emplace_back(work, w);
    work(0);
    for (std::thread& t : pool) t.join();
}

bool Swarm::inferAll() {
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    for (Duck& d : _ducks) {
        if (d.faulted) continue;
        ok = _onnx.runFloat(d.obs, d.act) && d.act.size() == 14 && ok;
    }
    _policyMicros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

void Swarm::actuateAndStep(bool actionsOk) {
    for (Duck& d : _ducks) {
        std::vector<float> targets = _defaultAct;                    // rad; a faulted duck holds the default pose
        if (!d.faulted) {
            if (!actionsOk || d.act.size() != 14) {
                d.faulted = true;
            } else {
                d.last = d.act;
                for (size_t i = 0; i < targets.size(); ++i) targets[i] = _defaultAct[i] + _scale[i] * d.act[i];
                for (auto& l : d.limbs) {
                    for (int i = 0; i < l.target.size(); ++i) l.target[i] = targets[l.actionIdx[i]] * 180.0 / kPi;
                    if (!l.gate.observe(l.target, (float)(_decimation * _simDt))) { d.faulted = true; break; }
                }
                if (d.faulted) targets = _defaultAct;
            }
        }
        for (size_t i = 0; i < d.actuator.size(); ++i) _d->ctrl[d.actuator[i]] = targets[i];
    }
    for (int s = 0; s < _decimation; ++s) mj_step(_m, _d);
    mj_kinematics(_m, _d);                                           // so xpos/xmat belong to the same state as qpos
    for (const Duck& d : _ducks) if (!std::isfinite(_d->qpos[d.freeQpos + 2])) { reset(); return; }
    scanContacts();
    // a duck that has been on its side for a fifth of a second, or whose targets the gate refused, is stood up again somewhere clear
    for (size_t k = 0; k < _ducks.size(); ++k) {
        Duck& d = _ducks[k];
        d.downPeriods = uprightness((int)k) < 0.5 ? d.downPeriods + 1 : 0;
        if (d.downPeriods >= 10 || d.faulted) respawn((int)k);
    }
}

// A spot inside the walls that is at least 0.6 m from every other duck (the farthest one tried, if none is).
glm::dvec2 Swarm::freeSpot(int exceptDuck) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double radius = _half - 0.4;
    glm::dvec2 best(0.0);
    double bestGap = -1.0;
    for (int tries = 0; tries < 200; ++tries) {
        const double r = radius * std::sqrt(uni(_rng)), a = 2.0 * kPi * uni(_rng);
        const glm::dvec2 p(r * std::cos(a), r * std::sin(a));
        double gap = 1e9;
        for (size_t j = 0; j < _ducks.size(); ++j)
            if ((int)j != exceptDuck) gap = std::min(gap, glm::length(p - glm::dvec2(pos((int)j))));
        if (!clearOfObstacles(p, 0.35)) continue;
        if (gap > bestGap) { bestGap = gap; best = p; }
        if (gap >= 0.6) break;
    }
    return best;
}

void Swarm::respawn(int k) {
    Duck& d = _ducks[k];
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const glm::dvec2 p = freeSpot(k);
    const double yaw = (2.0 * uni(_rng) - 1.0) * kPi;
    double* q = _d->qpos + d.freeQpos;
    q[0] = p.x; q[1] = p.y; q[2] = _initHeight;
    q[3] = std::cos(yaw / 2.0); q[4] = 0.0; q[5] = 0.0; q[6] = std::sin(yaw / 2.0);
    for (size_t i = 0; i < d.obsQpos.size(); ++i) _d->qpos[d.obsQpos[i]] = _defaultObs[i];
    const int freeDof = _m->body_dofadr[d.trunk];
    for (int i = 0; i < 6; ++i) _d->qvel[freeDof + i] = 0.0;
    for (int dof : d.obsDof) _d->qvel[dof] = 0.0;
    for (size_t i = 0; i < d.actuator.size(); ++i) _d->ctrl[d.actuator[i]] = _defaultAct[i];
    std::fill(d.last.begin(), d.last.end(), 0.f);
    d.headingTarget = yaw;
    d.faulted = false;
    d.downPeriods = 0;
    for (auto& l : d.limbs) l.gate.clearFault();
    mj_kinematics(_m, _d);
}

void Swarm::scanContacts() {
    bool duckHit = false, wallHit = false, obstacleHit = false;
    for (int i = 0; i < _d->ncon; ++i) {
        const mjContact& c = _d->contact[i];
        const int a = _geomDuck[c.geom1], b = _geomDuck[c.geom2];
        if (a >= 0 && b >= 0 && a != b) {
            duckHit = true;
            if (!_touched[a][b]) { _touched[a][b] = _touched[b][a] = 1; ++_stats.duckPairs; }
        } else if ((a >= 0 && b == -2) || (b >= 0 && a == -2)) {
            wallHit = true;
        } else if ((a >= 0 && b == -3) || (b >= 0 && a == -3)) {
            obstacleHit = true;
        }
    }
    _stats.duckContactPeriods += duckHit;
    _stats.wallContactPeriods += wallHit;
    _stats.obstacleContactPeriods += obstacleHit;
    for (size_t i = 0; i < _ducks.size(); ++i)
        for (size_t j = i + 1; j < _ducks.size(); ++j)
            _stats.closest = std::min(_stats.closest, glm::length(glm::dvec2(pos((int)i)) - glm::dvec2(pos((int)j))));
}

void Swarm::policyStep() {
    using clock = std::chrono::steady_clock;
    auto us = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    const auto t0 = clock::now();
    observe();
    const auto t1 = clock::now();
    const bool ok = inferAll();
    const auto t2 = clock::now();
    actuateAndStep(ok);
    const auto t3 = clock::now();
    _tObserve += us(t0, t1); _tInfer += us(t1, t2); _tStep += us(t2, t3);
    ++_periods;
}

void Swarm::startThread() {
    if (_physicsThread.joinable()) return;
    _run = true;
    _inferenceThread = std::thread([this] { inferenceLoop(); });
    _physicsThread = std::thread([this] { physicsLoop(); });
}

void Swarm::stopThread() {
    _run = false;
    _obsReady.notify_all();
    _actReady.notify_all();
    if (_physicsThread.joinable()) _physicsThread.join();
    if (_inferenceThread.joinable()) _inferenceThread.join();
}

// The inference thread: wait for the observations, run the policy once per duck, hand the actions back.
void Swarm::inferenceLoop() {
    std::vector<float> obs, in, out;
    std::vector<float> acts;
    while (_run) {
        long seq;
        {
            std::unique_lock<std::mutex> lock(_mail);
            _obsReady.wait(lock, [&] { return !_run || _obsSeq != _inferSeq; });
            if (!_run) return;
            obs = _mailObs;
            seq = _inferSeq = _obsSeq;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const size_t od = _ducks[0].obs.size();
        acts.assign(_ducks.size() * 14, 0.f);
        bool ok = true;
        for (size_t k = 0; k < _ducks.size(); ++k) {
            in.assign(obs.begin() + k * od, obs.begin() + (k + 1) * od);
            ok = _onnx.runFloat(in, out) && out.size() == 14 && ok;
            if (out.size() == 14) std::copy(out.begin(), out.end(), acts.begin() + k * 14);
        }
        _policyMicros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lock(_mail);
            _mailAct = acts;
            _mailOk = ok;
            _actSeq = seq;
        }
        _actReady.notify_all();
    }
}

// The physics thread: real time, one policy period (four 5 ms steps) at a time, for the whole swarm.
void Swarm::physicsLoop() {
    using clock = std::chrono::steady_clock;
    const double period = _decimation * _simDt;
    auto last = clock::now();
    double accum = 0.0;
    while (_run) {
        const auto now = clock::now();
        accum += std::min(std::chrono::duration<double>(now - last).count(), 0.1);
        last = now;
        while (accum >= period && _run) {
            accum -= period;
            if (paused) continue;
            long seq = 0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                observe();
                std::lock_guard<std::mutex> mail(_mail);
                _mailObs.clear();
                for (const Duck& d : _ducks) _mailObs.insert(_mailObs.end(), d.obs.begin(), d.obs.end());
                seq = ++_obsSeq;
            }
            _obsReady.notify_all();
            bool ok = true;
            std::vector<float> acts;
            {
                std::unique_lock<std::mutex> lock(_mail);
                // twelve answers take about a millisecond; if they are late this period keeps the last actions
                if (_actReady.wait_for(lock, std::chrono::milliseconds(8), [&] { return _actSeq == seq || !_run; }) && _actSeq == seq) {
                    ok = _mailOk;
                    acts = _mailAct;
                } else {
                    ok = false;
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (acts.empty()) {                                      // late: hold what each had
                for (Duck& d : _ducks) d.act = d.last;
                ok = true;
            } else {
                for (size_t k = 0; k < _ducks.size(); ++k) std::copy(acts.begin() + k * 14, acts.begin() + (k + 1) * 14, _ducks[k].act.begin());
            }
            actuateAndStep(ok);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
