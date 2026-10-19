#include "ofApp.h"
#include <map>

namespace {
const double kDeg = 180.0 / 3.14159265358979;
const double kPi = 3.14159265358979;
const float kFootContactZ = 0.040f;     // a foot's sphere centre this low is on the floor (its radius is 0.031)
}

bool ofApp::connectClient(bool realtime) {
    QuadrupedClient::Config c;
    c.realtime = realtime;
    if (render.enabled() && !_opt.check) { c.spawnX = -0.9; }   // a recording starts at the left of the scene, walking right
    if (!_client.connect(c)) return false;
    // connect() returns at once with a worker thread; the model is ready when it reports connected
    for (int waited = 0; !_client.isConnected() && waited < 1000; ++waited) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return _client.isConnected();
}

void ofApp::setup() {
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    ofSetSphereResolution(12);
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) _legs[l] = &_legState[l];
    _client.setup(_legs);

    if (_opt.parity) {
        if (!connectClient(false) || !_brain.setup()) { ofLogError("ofApp") << "setup failed"; ofExit(1); return; }
        runParity();
        return;
    }
    if (_opt.check) { runCheck(); return; }

    const bool recording = render.enabled();
    if (!connectClient(!recording) || !_brain.setup()) { ofLogError("ofApp") << "setup failed"; ofExit(1); return; }

    // The robot is drawn from its visual meshes (ofxRobot's RobotMesh loads the binary STLs the model was made with, one per link,
    // in the link's own frame), lit by a dome of light; a colour per kind of link: a light shell, ANYbotics' warm yellow thighs,
    // graphite shanks, black feet and hips.
    _font.load(OF_TTF_MONO, recording ? std::max(12, render.height / 56) : 13);
    _labels.setFont(_font);
    _dome.setup();
    const mjModel* m = _client.model();
    const ofFloatColor grey = brandColour("anymal_d");   // ofxRobot's make colour: mid-grey for anything it does not know
    const ofFloatColor shell(grey.r * 1.5f, grey.g * 1.5f, grey.b * 1.5f + 0.03f);
    for (int b = 1; b < m->nbody; ++b) {
        const std::string name = mj_id2name(m, mjOBJ_BODY, b);
        Part p;
        if (!robotmesh::loadBinaryStl(ofToDataPath("anymal_d/meshes/visual/" + name + ".stl", true), p.mesh)) continue;
        robotmesh::smoothNormals(p.mesh);
        p.body = b;
        const auto ends = [&](const char* s) { const std::string e(s); return name.size() >= e.size() && name.compare(name.size() - e.size(), e.size(), e) == 0; };
        ofFloatColor c = shell;
        float shine = 55.f, spec = 0.35f;
        if (ends("THIGH")) { c = ofFloatColor(0.97f, 0.66f, 0.10f); shine = 70.f; spec = 0.45f; }
        else if (ends("SHANK")) { c = ofFloatColor(0.20f, 0.21f, 0.24f); spec = 0.5f; }
        else if (ends("HIP")) { c = ofFloatColor(0.13f, 0.14f, 0.16f); }
        else if (ends("FOOT")) { c = ofFloatColor(0.07f, 0.07f, 0.08f); spec = 0.15f; shine = 20.f; }
        p.material.setDiffuseColor(c);
        p.material.setAmbientColor(c * 0.55f);
        p.material.setSpecularColor(ofFloatColor(spec));
        p.material.setShininess(shine);
        _parts.push_back(p);
    }
    for (int g = 0; g < m->ngeom; ++g) {      // each foot's collision sphere
        if (m->geom_type[g] != mjGEOM_SPHERE) continue;
        const std::string body = mj_id2name(m, mjOBJ_BODY, m->geom_bodyid[g]);
        for (int l = 0; l < QuadrupedClient::kLegs; ++l)
            if (body == _client.legs()[l].toolBody) _footGeoms[l].push_back(g);
    }
    _showKinematics = _opt.kin;
    _showCollision = _opt.collision;
    _grid.setMode(OF_PRIMITIVE_LINES);
    _cam.setNearClip(0.02f);
    _cam.setFarClip(150.f);
    _cam.setFov(38.f);

    if (_opt.publishPort > 0) {   // each leg's RobotState, as JSON over TCP and UDP
        for (int l = 0; l < QuadrupedClient::kLegs; ++l) _pub[l].setup(_legs[l], _opt.publishPort + 2 * l, _opt.publishPort + 2 * l + 1);
        _publishing = true;
    }
    const QuadrupedClient::Sense s = _client.sense();
    _focus = glm::vec3(s.pos);
    _focusInit = true;
    if (recording) {
        render.realtime = false;     // --record and --render both step the world by the frame time: the same video on any machine
        render.warmup = 1.0;         // the robot stands and settles for a second before the first frame
        _rec.setup(render, 30.0);
        _azOffset = -42.f; _el = 16.f; _dist = 3.0f;
    }
}

// The headless walk. Offline: the world is stepped by this loop, one 5 ms physics step at a time, the brain asked at every policy
// period, so it gives the same answer on any machine. Three things are judged, the first two about ofxRobot's streaming-client contract:
//   1. streaming is off by default and the legs hold the stand pose;
//   2. a bad target (a NaN) stops streaming, holds the stand pose, and stays stopped until the operator clears it and turns it on;
//   3. with streaming on, the policy walks: it stays up, gets somewhere, and ofxRobot's forward kinematics agrees with the simulation.
void ofApp::runCheck() {
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) _legs[l] = &_legState[l];
    _client.setup(_legs);
    QuadrupedClient::Config cfg;
    cfg.realtime = _opt.threaded;
    cfg.autoReset = false;               // the check judges a fall, it does not hide it
    if (!_client.connect(cfg) || !_brain.setup()) { ofLogError("ofApp") << "setup failed"; ofExit(1); return; }
    for (int waited = 0; !_client.isConnected() && waited < 1000; ++waited) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool threaded = _opt.threaded;
    bool ok = true;
    auto step = [&](double seconds, bool tickBrain) {   // run the loop for that long (simulated; real time with --threads)
        const int n = (int)std::lround(seconds / QuadrupedClient::kDt);
        for (int i = 0; i < n; ++i) {
            if (_client.consumeJustConnected()) { _client.update(); for (RobotState* st : _legs) st->targetJoints = st->currentJoints; }
            if (tickBrain) _brain.tick(_client.sense(), _legs);
            _client.update();
            if (threaded) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            else _client.stepOffline();
        }
    };
    auto standHeld = [&]() {   // every command is the stand pose
        double worst = 0.0;
        const std::vector<float> d = {0.f, 0.4f, -0.8f, 0.f, -0.4f, 0.8f, 0.f, 0.4f, -0.8f, 0.f, -0.4f, 0.8f};
        for (int l = 0; l < QuadrupedClient::kLegs; ++l)
            for (int j = 0; j < QuadrupedClient::kLegJoints; ++j)
                worst = std::max(worst, std::fabs(_legs[l]->commandedJoints[j] - d[l * 3 + j] * kDeg));
        return worst;
    };

    // 1. Streaming off by default: the brain asks, the legs hold.
    _brain.setCommand(0.5f, 0.f, 0.f);
    const bool offByDefault = !_client.isStreaming();
    step(1.0, true);
    const double holdErr = standHeld();
    const QuadrupedClient::Sense hold = _client.sense();
    const bool holdOk = offByDefault && holdErr < 0.01 && hold.pos.z > 0.45 && hold.rot[2][2] > 0.95 && std::fabs(hold.pos.x) < 0.05;
    ofLogNotice("quadruped") << (holdOk ? "ok" : "FAIL") << ": streaming off by default: " << (offByDefault ? "off" : "ON") << ", the legs hold the stand pose (largest command error "
                             << holdErr << " deg, body height " << hold.pos.z << " m, moved " << hold.pos.x << " m although the brain asked for 0.5 m/s)";
    ok = ok && holdOk;

    // 2. A bad target stops streaming, and nothing resumes it by itself.
    _client.setStreaming(true);
    _brain.reset();
    step(0.6, true);
    const bool streamingBefore = _client.isStreaming();
    _legs[0]->targetJoints[1] = std::nan("");               // an app (or a policy) asserts a NaN
    _legs[0]->hasTarget = true;
    for (int i = 0; i < 4; ++i) { _client.update(); if (threaded) std::this_thread::sleep_for(std::chrono::milliseconds(5)); else _client.stepOffline(); }
    const bool faulted = _client.hasFault() && !_client.isStreaming();
    const std::string why = _client.faultMessage();
    step(0.5, true);                                         // the brain keeps asking: the fault holds, the legs go back to the stand pose
    _client.setStreaming(true);                              // refused while the fault stands
    const bool staysOff = _client.hasFault() && !_client.isStreaming();
    _client.clearFault();
    step(0.05, true);
    const bool clearedNotResumed = !_client.hasFault() && !_client.isStreaming();   // the operator cleared it: streaming is still off
    _client.setStreaming(true);
    const bool resumed = _client.isStreaming();              // and only now, on request, back on
    const bool faultOk = streamingBefore && faulted && staysOff && clearedNotResumed && resumed;
    ofLogNotice("quadruped") << (faultOk ? "ok" : "FAIL") << ": a NaN target: streaming " << (streamingBefore ? "was on" : "was OFF") << ", then " << (faulted ? "stopped with fault '" + why + "'" : "did NOT stop")
                             << "; refused while faulted: " << (staysOff ? "yes" : "NO") << "; cleared by the operator, streaming still off: " << (clearedNotResumed ? "yes" : "NO")
                             << "; turned on again on request: " << (resumed ? "yes" : "NO");
    ok = ok && faultOk;

    // 3. The walk: start again from standing.
    _client.setStreaming(false);
    _client.resetRobot();
    step(0.1, false);
    _client.setStreaming(true);
    _brain.reset();
    const float command = 0.5f;
    _brain.setCommand(command, 0.f, 0.f);
    const QuadrupedClient::Sense s0 = _client.sense();
    const glm::dvec3 start = s0.pos;
    const glm::dvec3 heading = glm::normalize(glm::dvec3(s0.rot[0][0], s0.rot[0][1], 0.0));
    double worstUp = 1.0, worstFk = 0.0, policyUs = 0.0, torqueMax = 0.0, minHeight = 1e9, maxHeight = 0.0;
    std::map<std::string, double> fkAtRest, fkWorst;
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) fkAtRest[_client.legs()[l].name] = glm::length(_client.toolFk(l) - _client.toolSim(l)) * 1000.0;
    int steps = 0;
    const double t0 = s0.simTime;
    bool up = true;
    double nextLog = 5.0;
    for (;; ++steps) {
        step(threaded ? QuadrupedClient::kDt : 0.02, true);  // one policy period (one physics step when the worker thread runs it in real time)
        const QuadrupedClient::Sense s = _client.sense();
        if (s.simTime - t0 >= _opt.checkSeconds) break;
        worstUp = std::min(worstUp, s.rot[2][2]);
        worstFk = std::max(worstFk, _client.maxFkErrorMm());
        for (int l = 0; l < QuadrupedClient::kLegs; ++l)
            fkWorst[_client.legs()[l].name] = std::max(fkWorst[_client.legs()[l].name], glm::length(_client.toolFk(l) - _client.toolSim(l)) * 1000.0);
        policyUs += _brain.policyMicros();
        torqueMax = std::max(torqueMax, _client.torqueMax());
        minHeight = std::min(minHeight, s.pos.z);
        maxHeight = std::max(maxHeight, s.pos.z);
        if (s.simTime - t0 >= nextLog) {
            nextLog += 5.0;
            ofLogNotice("quadruped") << "  t=" << s.simTime - t0 << " s  position (" << s.pos.x << ", " << s.pos.y << ")  speed in its own forward " << s.linVelBody.x
                                     << " m/s  height " << s.pos.z;
        }
        if (_client.hasFault() || s.rot[2][2] < 0.5 || s.pos.z < 0.25) { up = false; break; }
    }
    const QuadrupedClient::Sense end = _client.sense();
    const double forward = glm::dot(end.pos - start, heading);
    const double seconds = end.simTime - t0;
    const bool walkOk = up && forward > 0.5 * command * _opt.checkSeconds && worstFk < 0.01 && !_client.hasFault();
    ofLogNotice("quadruped") << (walkOk ? "ok" : "FAIL") << ": " << (up ? "stayed up" : "FELL") << " for " << seconds << " s, walked " << forward << " m forward ("
                             << (seconds > 0 ? forward / seconds : 0.0) << " m/s for a command of " << command << "), body height " << minHeight << " to " << maxHeight
                             << " m, worst tilt (up = 1) " << worstUp << ", largest joint torque " << torqueMax << " N m, ofxRobot FK vs simulation: worst " << worstFk
                             << " mm, policy " << (steps ? policyUs / steps : 0.0) << " us/step, actuator network " << _client.actuatorMicros() << " us/call (12 joints), "
                             << "largest command step " << _client.largestStepDeg() << " deg in 5 ms (speed cap " << (_client.speedLimited() ? "HIT" : "not reached") << ", limits clamped mask "
                             << _client.clampedMask() << "), safety gate " << (_client.hasFault() ? "FAULT: " + _client.faultMessage() : std::string("ok"));
    for (int l = 0; l < QuadrupedClient::kLegs; ++l)
        ofLogNotice("quadruped") << "  " << _client.legs()[l].name << ": FK vs simulation " << fkAtRest[_client.legs()[l].name] << " mm at rest, worst "
                                 << fkWorst[_client.legs()[l].name] << " mm walking";
    ok = ok && walkOk;
    ofLogNotice("quadruped") << (ok ? "PASS" : "FAIL");
    _client.close();
    ofExit(ok ? 0 : 1);
}

// The two networks, through ofxOnnx, against what PyTorch said (tools/export_models.py stored both).
void ofApp::runParity() {
    const ofJson ref = ofLoadJson("anymal_d/reference.json");
    double policyWorst = 0.0, actuatorWorst = 0.0;
    int policyCases = 0, actuatorSteps = 0;
    std::vector<float> act;
    for (size_t i = 0; i < ref["policy"]["obs"].size(); ++i) {
        const std::vector<float> obs = ref["policy"]["obs"][i].get<std::vector<float>>();
        const std::vector<float> want = ref["policy"]["actions"][i].get<std::vector<float>>();
        if (!_brain.runPolicy(obs, act) || act.size() != want.size()) { ofLogNotice("quadruped") << "FAIL: the policy did not answer"; ofExit(1); return; }
        for (size_t k = 0; k < want.size(); ++k) policyWorst = std::max(policyWorst, (double)std::fabs(act[k] - want[k]));
        ++policyCases;
    }
    const int n = ref["actuator"]["n"].get<int>();
    std::vector<float> h(2 * n * 8, 0.f), c(2 * n * 8, 0.f), tau;
    for (const auto& step : ref["actuator"]["steps"]) {
        std::vector<float> x;
        for (const auto& row : step["x"]) for (const auto& v : row) x.push_back(v.get<float>());
        const std::vector<float> want = step["torque"].get<std::vector<float>>();
        if (!_client.runActuator(n, x, h, c, tau) || (int)tau.size() != n) { ofLogNotice("quadruped") << "FAIL: the actuator network did not answer"; ofExit(1); return; }
        for (int k = 0; k < n; ++k) actuatorWorst = std::max(actuatorWorst, (double)std::fabs(tau[k] - want[k]));
        ++actuatorSteps;
    }
    const double tolerance = 1e-4;
    const bool ok = policyWorst < tolerance && actuatorWorst < tolerance;
    ofLogNotice("quadruped") << (ok ? "PASS" : "FAIL") << ": ofxOnnx vs PyTorch, policy " << policyCases << " observations, worst |difference| " << policyWorst
                             << "; actuator network " << actuatorSteps << " steps x " << n << " joints (LSTM state carried), worst |difference| " << actuatorWorst
                             << " N m; tolerance " << tolerance;
    ofExit(ok ? 0 : 1);
}

void ofApp::exit() {
    _rec.finish();
    for (auto& p : _pub) p.close();
    _client.close();
}

// The recording's walk, by the simulation's clock, within about three metres of the middle: stand, walk away, turn round on the spot,
// walk back, sidestep away from the camera, back up, stand.
void ofApp::scriptCommand(double t) {
    struct Segment { double until; float vx, vy, wz; };
    static const Segment script[] = {
        {3.0, 0.f, 0.f, 0.f},          // stand (the first second is the warm-up, not recorded)
        {6.6, 0.5f, 0.f, 0.f},         // walk away (1.8 m)
        {7.6, 0.f, 0.f, 0.f},
        {11.8, 0.f, 0.f, 0.8f},        // turn round on the spot (about 190 degrees)
        {15.4, 0.5f, 0.f, 0.f},        // walk back
        {17.4, 0.f, -0.4f, 0.f},       // sidestep to its right (away from a camera in front of the path)
        {20.4, -0.4f, 0.f, 0.f},       // back up
        {21.4, 0.f, 0.f, 0.f},
        {23.9, 0.4f, 0.f, 0.f},        // walk on
        {27.5, 0.f, 0.f, 0.8f},        // and turn
        {1e9, 0.f, 0.f, 0.f},          // stand
    };
    for (const Segment& s : script)
        if (t < s.until) { const float k = (float)(_opt.walkSpeed / 0.5); _vx = s.vx * k; _vy = s.vy * k; _wz = s.wz; return; }
}

// One round of the loop: the brain reads what the controller reported and writes joint targets into the legs' RobotStates; the
// client takes them and hands back the measured joints; the publishers copy the states.
void ofApp::exchange() {
    if (_client.consumeJustConnected()) {   // seed the target from where the legs are, so the first command is not a jump
        _client.update();
        for (RobotState* st : _legs) { st->targetJoints = st->currentJoints; st->hasTarget = false; }
        _brain.reset();
    }
    if (_client.resets() != _lastResets) { _lastResets = _client.resets(); _brain.reset(); }
    const QuadrupedClient::Sense s = _client.sense();
    if (!_streamed && !_opt.hold && s.simTime >= _streamAt) {   // the robot has settled: this example turns streaming on by itself (a hardware app would not)
        _client.setStreaming(true);
        _brain.reset();
        _streamed = true;
    }
    _brain.setCommand(_vx, _vy, _wz);
    _brain.tick(s, _legs);
    _client.update();
    if (_publishing) {
        for (int l = 0; l < QuadrupedClient::kLegs; ++l) {   // the foot, by ofxRobot's forward kinematics, is each leg's "tcp"
            RobotState& st = *_legs[l];
            const glm::dmat4 cur = _client.footFk(l, st.currentJoints);
            st.tcpCurrentPos = glm::vec3(cur[3]);
            st.tcpCurrentRot = glm::quat_cast(glm::mat3(cur));
            const glm::dmat4 tgt = _client.footFk(l, st.targetJoints);
            st.tcpTargetPos = glm::vec3(tgt[3]);
            st.tcpTargetRot = glm::quat_cast(glm::mat3(tgt));
            _pub[l].update();
        }
    }
}

// Where each foot came down: a mark that fades, and which feet are on the floor now.
void ofApp::trackFeet(const QuadrupedClient::Sense& s) {
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) {
        bool on = false;
        glm::vec3 p(0.f);
        for (int g : _footGeoms[l]) {
            const glm::vec3 c(_geomPose[g][3]);
            if (c.z < kFootContactZ) { on = true; p = c; }
        }
        if (on && !_contact[l]) _steps.push_back({glm::vec2(p.x, p.y), s.simTime, l});
        _contact[l] = on;
    }
    while (!_steps.empty() && s.simTime - _steps.front().t > 4.0) _steps.pop_front();
}

void ofApp::update() {
    if (_opt.check || _opt.parity) return;
    const double dt = _rec.active() ? _rec.dt() : std::min((double)ofGetLastFrameTime(), 0.1);
    if (_rec.active()) {   // recording: the world is stepped here, by whole 5 ms physics steps of the frame's time, on this thread
        _carry += dt / QuadrupedClient::kDt;
        while (_carry >= 1.0) {
            scriptCommand(_client.sense().simTime);
            exchange();
            _client.stepOffline();
            _carry -= 1.0;
        }
    } else {
        exchange();
        const QuadrupedClient::Sense s = _client.sense();
        if (dt > 0.0 && s.simTime >= _lastSim) _realtime += 0.1 * ((s.simTime - _lastSim) / dt - _realtime);
        _lastSim = s.simTime;
    }

    // The follow camera: damped, three quarters from the front. The focus follows the trunk; the camera's heading follows the robot's, slowly.
    const QuadrupedClient::Sense s = _client.sense();
    const double yaw = std::atan2(s.rot[0][1], s.rot[0][0]);
    if (!_focusInit) { _focus = glm::vec3(s.pos); _camHeading = yaw; _focusInit = true; }
    const glm::dvec3 vWorld = s.rot * s.linVelBody;
    const glm::vec3 lead = glm::vec3(s.pos + 0.35 * glm::dvec3(vWorld.x, vWorld.y, 0.0));   // aim a little ahead, so the damped follow does not trail
    _focus += (lead - _focus) * (float)(1.0 - std::exp(-dt / 0.30));
    double dh = std::atan2(std::sin(yaw - _camHeading), std::cos(yaw - _camHeading));
    _camHeading += dh * (1.0 - std::exp(-dt / 1.2));
    _client.bodyPoses(_bodyPose);
    _client.geomPoses(_geomPose);
    trackFeet(s);
}

void ofApp::placeCamera(const glm::vec3& target, float azDeg, float elDeg, float dist) {
    const float az = glm::radians(azDeg), el = glm::radians(elDeg);
    _cam.setPosition(target + glm::vec3(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)) * dist);
    _cam.lookAt(target, glm::vec3(0.f, 0.f, 1.f));
}

// The robot: every link's mesh at the pose the last physics step published.
void ofApp::drawRobot() {
    ofEnableLighting();
    _dome.enable();
    for (Part& p : _parts) {
        ofPushMatrix();
        ofMultMatrix(_bodyPose[p.body]);
        p.material.begin();
        p.mesh.draw();
        p.material.end();
        ofPopMatrix();
    }
    _dome.disable();
    ofDisableLighting();
}

// The floor: a dark disc that fades into the distance, a grid that fades with distance, a soft contact shadow under the robot, and
// where the feet came down.
void ofApp::drawFloor(const QuadrupedClient::Sense& s) {
    ofEnableAlphaBlending();
    // the disc: opaque near the robot, transparent far away, so the floor melts into the horizon
    static ofVboMesh disc;   // (several ofMesh::draw calls in one frame can all come out with the last one's geometry on this OF build, see LabelBatch: so ofVboMesh, one per drawable, a fresh one each frame when it changes)
    if (disc.getNumVertices() == 0) {
        disc.setMode(OF_PRIMITIVE_TRIANGLES);
        const int n = 72;
        const float r1 = 6.f, r2 = 22.f;
        disc.addVertex(glm::vec3(0.f)); disc.addColor(ofFloatColor(0.20f, 0.215f, 0.25f, 1.f));
        for (int i = 0; i < n; ++i) {
            const float a = 2.f * (float)kPi * i / n;
            disc.addVertex(glm::vec3(r1 * std::cos(a), r1 * std::sin(a), 0.f)); disc.addColor(ofFloatColor(0.20f, 0.215f, 0.25f, 1.f));
            disc.addVertex(glm::vec3(r2 * std::cos(a), r2 * std::sin(a), 0.f)); disc.addColor(ofFloatColor(0.20f, 0.22f, 0.27f, 0.f));
        }
        for (int i = 0; i < n; ++i) {
            const unsigned a = 1 + 2 * i, b = 2 + 2 * i, c = 1 + 2 * ((i + 1) % n), d = 2 + 2 * ((i + 1) % n);
            disc.addTriangle(0, a, c);
            disc.addTriangle(a, b, d);
            disc.addTriangle(a, d, c);
        }
    }
    ofPushMatrix();
    ofTranslate(_focus.x, _focus.y, -0.002f);
    ofSetColor(255);
    disc.draw();
    ofPopMatrix();

    // the grid, 0.5 m, fixed in the world (snapped to the lines near the focus), each segment as bright as it is close
    _grid = ofVboMesh();
    _grid.setMode(OF_PRIMITIVE_LINES);
    const float step = 0.5f, R = 9.f;
    const float cx = std::round(_focus.x / step) * step, cy = std::round(_focus.y / step) * step;
    auto shade = [&](float x, float y) {
        const float d = std::hypot(x - _focus.x, y - _focus.y) / R;
        const float a = std::max(0.f, 1.f - d);
        return ofFloatColor(0.75f, 0.8f, 0.9f, 0.20f * a * a);
    };
    for (float k = -R; k <= R + 1e-3f; k += step) {
        for (float t = -R; t < R - 1e-3f; t += step) {
            _grid.addVertex(glm::vec3(cx + t, cy + k, 0.f));        _grid.addColor(shade(cx + t, cy + k));
            _grid.addVertex(glm::vec3(cx + t + step, cy + k, 0.f)); _grid.addColor(shade(cx + t + step, cy + k));
            _grid.addVertex(glm::vec3(cx + k, cy + t, 0.f));        _grid.addColor(shade(cx + k, cy + t));
            _grid.addVertex(glm::vec3(cx + k, cy + t + step, 0.f)); _grid.addColor(shade(cx + k, cy + t + step));
        }
    }
    ofSetLineWidth(1.f);
    _grid.draw();

    // the contact shadow: a soft dark ellipse under the trunk (a fan of rings, the alpha falling to nothing at the rim)
    {
        const double yaw = std::atan2(s.rot[0][1], s.rot[0][0]);
        const float cy2 = (float)std::cos(yaw), sy = (float)std::sin(yaw);
        ofVboMesh sh;
        sh.setMode(OF_PRIMITIVE_TRIANGLES);
        const int n = 48, rings = 5;
        sh.addVertex(glm::vec3((float)s.pos.x, (float)s.pos.y, 0.003f));
        sh.addColor(ofFloatColor(0.f, 0.f, 0.f, 0.78f));
        for (int r = 1; r <= rings; ++r) {
            const float f = (float)r / rings, a = 0.78f * (1.f - f) * (1.f - f);
            for (int i = 0; i < n; ++i) {
                const float t = 2.f * (float)kPi * i / n;
                const float ex = 0.80f * f * std::cos(t), ey = 0.42f * f * std::sin(t);
                sh.addVertex(glm::vec3((float)s.pos.x + ex * cy2 - ey * sy, (float)s.pos.y + ex * sy + ey * cy2, 0.003f));
                sh.addColor(ofFloatColor(0.f, 0.f, 0.f, a));
            }
        }
        for (int i = 0; i < n; ++i) sh.addTriangle(0, 1 + i, 1 + (i + 1) % n);
        for (int r = 0; r < rings - 1; ++r)
            for (int i = 0; i < n; ++i) {
                const unsigned a0 = 1 + r * n + i, a1 = 1 + r * n + (i + 1) % n, b0 = 1 + (r + 1) * n + i, b1 = 1 + (r + 1) * n + (i + 1) % n;
                sh.addTriangle(a0, b0, b1);
                sh.addTriangle(a0, b1, a1);
            }
        ofSetColor(255);
        sh.draw();
    }

    // where the feet came down, and which are down now
    static const ofColor tint[2] = {ofColor(255, 190, 70), ofColor(90, 200, 255)};   // the two diagonal pairs of a trot
    for (const Step& st : _steps) {
        const float age = (float)(s.simTime - st.t);
        const float a = std::max(0.f, 1.f - age / 4.f);
        ofColor c = tint[(st.leg == 0 || st.leg == 3) ? 0 : 1];
        ofSetColor(c, (int)(150.f * a));
        ofDrawCircle(glm::vec3(st.pos.x, st.pos.y, 0.006f), 0.030f);
    }
    ofNoFill();
    ofSetLineWidth(2.f);
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) {
        if (!_contact[l]) continue;
        for (int g : _footGeoms[l]) {
            const glm::vec3 c(_geomPose[g][3]);
            ofColor col = tint[(l == 0 || l == 3) ? 0 : 1];
            ofSetColor(col, 220);
            ofDrawCircle(glm::vec3(c.x, c.y, 0.008f), 0.055f);
        }
    }
    ofSetLineWidth(1.f);
    ofFill();
}

// The shapes the physics collides with: spheres on the feet, boxes, cylinders and a capsule for the body.
void ofApp::drawCollision() {
    const mjModel* m = _client.model();
    ofSetColor(70, 220, 255, 110);
    for (int g = 0; g < m->ngeom; ++g) {
        const int t = m->geom_type[g];
        if (t == mjGEOM_MESH || t == mjGEOM_PLANE) continue;
        const double* s = m->geom_size + 3 * g;
        ofPushMatrix();
        ofMultMatrix(_geomPose[g]);
        if (t == mjGEOM_SPHERE) ofDrawSphere(glm::vec3(0.f), (float)s[0]);
        else if (t == mjGEOM_BOX) ofDrawBox(glm::vec3(0.f), 2.f * (float)s[0], 2.f * (float)s[1], 2.f * (float)s[2]);
        else if (t == mjGEOM_CYLINDER || t == mjGEOM_CAPSULE) {
            ofPushMatrix();
            ofRotateXDeg(90.f);                                  // ofDrawCylinder stands along y; MuJoCo's lies along z
            ofDrawCylinder(glm::vec3(0.f), (float)s[0], 2.f * (float)s[1]);
            ofPopMatrix();
            if (t == mjGEOM_CAPSULE) {
                ofDrawSphere(glm::vec3(0.f, 0.f, (float)s[1]), (float)s[0]);
                ofDrawSphere(glm::vec3(0.f, 0.f, -(float)s[1]), (float)s[0]);
            }
        }
        ofPopMatrix();
    }
}

// ofxRobot's view of the robot: each leg's joints from RobotKinematics + ArmIk forward kinematics, and where that puts the foot, next
// to where the simulation has it.
void ofApp::drawKinematics(const QuadrupedClient::Sense& s) {
    glm::mat4 T(1.f);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) T[c][r] = (float)s.rot[c][r];
    T[3] = glm::vec4(glm::vec3(s.pos), 1.f);
    ofDisableDepthTest();
    ofPushMatrix();
    ofMultMatrix(T);
    ofSetLineWidth(2.f);
    for (int l = 0; l < QuadrupedClient::kLegs; ++l) {
        const QuadrupedClient::Leg& leg = _client.legs()[l];
        const std::vector<glm::dmat4> frames = leg.ik.linkFrames(_legs[l]->currentJoints);
        glm::vec3 prev(frames[0][3]);   // the chain starts at the hip joint (not at the trunk's centre)
        ofSetColor(90, 225, 255);
        ofDrawSphere(prev, 0.010f);
        for (size_t i = 1; i < frames.size(); ++i) {
            const glm::vec3 p(frames[i][3]);
            ofDrawLine(prev, p);
            ofDrawSphere(p, 0.010f);
            prev = p;
        }
        const glm::vec3 tool(_client.toolFk(l));
        ofDrawLine(prev, tool);
        ofSetColor(255, 215, 60);
        ofDrawSphere(tool, 0.020f);                              // ofxRobot: the foot by forward kinematics
        ofSetColor(255, 70, 70, 220);
        ofDrawSphere(glm::vec3(_client.toolSim(l)), 0.011f);     // MuJoCo: the same body, from the simulation
    }
    ofSetLineWidth(1.f);
    ofPopMatrix();
    ofEnableDepthTest();
}

void ofApp::draw() {
    if (_opt.check || _opt.parity) return;
    if (_rec.active()) {
        if (_rec.beginFrame()) { drawScene(); _rec.endFrame(); }
        return;
    }
    drawScene();
    if (!_opt.shot.empty() && _client.sense().simTime > 4.0) {   // --shot: the window as it is, once the robot is walking
        ofImage img;
        img.grabScreen(0, 0, ofGetWidth(), ofGetHeight());
        img.save(_opt.shot);
        ofExit(0);
    }
}

void ofApp::drawScene() {
    const QuadrupedClient::Sense s = _client.sense();
    ofBackgroundGradient(ofColor(74, 84, 102), ofColor(20, 22, 28), OF_GRADIENT_LINEAR);
    glDepthMask(GL_TRUE);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);   // the window's depth buffer is not cleared by itself: last frame's floor would hide this one's
    ofEnableDepthTest();
    if (_opt.lock) placeCamera(glm::vec3(0.f, 0.4f, 0.36f), -90.f, 11.f, 3.7f);   // a fixed camera beside the path
    else placeCamera(glm::vec3(_focus.x, _focus.y, 0.32f), (float)(_camHeading * kDeg) + _azOffset, _el, _dist);
    _cam.begin();
    drawFloor(s);
    drawRobot();
    if (_showCollision) drawCollision();
    if (_showKinematics) drawKinematics(s);
    _cam.end();
    ofDisableDepthTest();
    if (_rec.active()) return;   // no text in a recording
    drawHud();
}

// A small panel in the corner: legible, and out of the robot's way.
void ofApp::drawHud() {
    const QuadrupedClient::Sense s = _client.sense();
    const glm::vec3 cmd = _brain.command();
    const double tilt = std::acos(std::min(1.0, std::max(-1.0, s.rot[2][2]))) * kDeg;
    float y = 10.f;
    const float lh = _font.getLineHeight() + 6.f;
    auto line = [&](const std::string& t, const ofColor& bg = ofColor(8, 10, 14, 150)) { _labels.add(t, 10.f, y, bg); y += lh; };
    line("ANYmal-D   MuJoCo + ofxOnnx + ofxRobot");
    line(std::string("streaming ") + (_client.isStreaming() ? "ON " : "off") + "   gate " + (_client.hasFault() ? "FAULT " + _client.faultMessage() : std::string("ok")),
         _client.hasFault() ? ofColor(170, 30, 30, 200) : (_client.isStreaming() ? ofColor(20, 90, 50, 190) : ofColor(110, 80, 10, 190)));
    line("command " + ofToString(cmd.x, 2) + " " + ofToString(cmd.y, 2) + " " + ofToString(cmd.z, 2) + "   speed " + ofToString(s.linVelBody.x, 2) + " m/s   tilt " + ofToString(tilt, 1) + " deg");
    line("policy " + ofToString(_brain.policyMicros(), 0) + " us   actuator net " + ofToString(_client.actuatorMicros(), 0) + " us   sim " + ofToString(_realtime, 2) + "x   FK " +
         ofToString(_client.maxFkErrorMm(), 3) + " mm   falls " + ofToString(_client.falls()));
    line("arrows walk/turn  , . strafe  space stop  s stream  x shove  r reset  k FK  g collision  f clear fault");
    _labels.draw();
}

void ofApp::keyPressed(int key) {
    switch (key) {
        case OF_KEY_UP:    _vx += 0.1f; break;
        case OF_KEY_DOWN:  _vx -= 0.1f; break;
        case OF_KEY_LEFT:  _wz += 0.2f; break;
        case OF_KEY_RIGHT: _wz -= 0.2f; break;
        case ',': _vy += 0.1f; break;
        case '.': _vy -= 0.1f; break;
        case ' ': _vx = _vy = _wz = 0.f; break;
        case 's': _client.setStreaming(!_client.isStreaming()); _streamed = true; _brain.reset(); break;
        case 'x': _client.push(0.f, 0.8f); break;
        case 'r': _client.resetRobot(); _steps.clear(); break;
        case 'p': _client.setPaused(!_client.paused()); break;
        case 'k': _showKinematics = !_showKinematics; break;
        case 'g': _showCollision = !_showCollision; break;
        case 'f': _client.clearFault(); break;
        default: break;
    }
    _vx = glm::clamp(_vx, -1.f, 1.f);
    _vy = glm::clamp(_vy, -1.f, 1.f);
    _wz = glm::clamp(_wz, -1.f, 1.f);
}

void ofApp::mouseDragged(int x, int y, int button) {
    const float dx = (float)(x - ofGetPreviousMouseX()), dy = (float)(y - ofGetPreviousMouseY());
    _azOffset -= dx * 0.3f;
    _el = glm::clamp(_el + dy * 0.3f, 2.f, 85.f);
}

void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY) {
    _dist = glm::clamp(_dist * std::pow(0.9f, scrollY), 0.8f, 30.f);
}
