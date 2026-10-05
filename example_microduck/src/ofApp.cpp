#include "ofApp.h"

void ofApp::setup() {
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    ofSetSphereResolution(10);

    if (!_duck.setup("microduck")) { ofLogError("ofApp") << "setup failed"; ofExit(1); return; }
    if (_check) { runCheck(); return; }

    // The model's meshes, as MuJoCo compiled them (so each already sits in its geom's frame): a triangle soup with
    // flat normals, one ofVboMesh per mesh.
    const mjModel* m = _duck.model();
    for (int i = 0; i < m->nmesh; ++i) {
        ofVboMesh mesh;
        mesh.setMode(OF_PRIMITIVE_TRIANGLES);
        const int v0 = m->mesh_vertadr[i], f0 = m->mesh_faceadr[i];
        for (int f = 0; f < m->mesh_facenum[i]; ++f) {
            glm::vec3 p[3];
            for (int c = 0; c < 3; ++c) {
                const int vi = m->mesh_face[3 * (f0 + f) + c];
                p[c] = glm::vec3(m->mesh_vert[3 * (v0 + vi)], m->mesh_vert[3 * (v0 + vi) + 1], m->mesh_vert[3 * (v0 + vi) + 2]);
            }
            glm::vec3 n = glm::cross(p[1] - p[0], p[2] - p[0]);
            n = glm::length(n) > 0.f ? glm::normalize(n) : glm::vec3(0, 0, 1);
            for (int c = 0; c < 3; ++c) { mesh.addVertex(p[c]); mesh.addNormal(n); }
        }
        _meshes.push_back(mesh);
    }

    _light.setPointLight();
    _light.setPosition(0.6f, -0.8f, 1.6f);
    _light.setDiffuseColor(ofFloatColor(1.f));
    _light.setAmbientColor(ofFloatColor(0.4f));

    _cam.setUpAxis(glm::vec3(0, 0, 1));
    _cam.setNearClip(0.005f);
    _cam.setFarClip(50.f);
    _focus = glm::vec3(_duck.trunkPos());
    _cam.setTarget(_focus);
    _cam.setPosition(_focus + glm::vec3(0.35f, -0.45f, 0.22f));
    _cam.lookAt(_focus, glm::vec3(0, 0, 1));
    if (!_check) _duck.startThread();
}

// A walk without a window: forward at 0.3 m/s. Judged on staying up, getting somewhere, and ofxRobot's forward
// kinematics agreeing with the simulation.
void ofApp::runCheck() {
    _duck.reset();
    _duck.setCommand(0.3f, 0.f, 0.f);
    const glm::dvec3 start = _duck.trunkPos();
    const glm::dmat3 R0 = _duck.trunkRot();
    const glm::dvec3 heading = glm::normalize(glm::dvec3(R0[0][0], R0[0][1], 0.0));
    const double yaw0 = _duck.heading();
    double worstUp = 1.0, worstFk = 0.0, policyUs = 0.0, worstDrift = 0.0;
    std::map<std::string, double> fkAtRest, fkWorst;
    for (const auto& l : _duck.limbs()) fkAtRest[l.name] = glm::length(_duck.toolFk(l) - _duck.toolSim(l)) * 1000.0;
    int steps = 0;
    const int total = (int)(_checkSeconds / 0.02);
    for (; steps < total; ++steps) {
        _duck.policyStep();
        worstUp = std::min(worstUp, _duck.uprightness());
        worstFk = std::max(worstFk, _duck.maxFkErrorMm());
        for (const auto& l : _duck.limbs()) fkWorst[l.name] = std::max(fkWorst[l.name], glm::length(_duck.toolFk(l) - _duck.toolSim(l)) * 1000.0);
        policyUs += _duck.policyMicros();
        worstDrift = std::max(worstDrift, std::fabs(std::atan2(std::sin(_duck.heading() - yaw0), std::cos(_duck.heading() - yaw0))) * 180.0 / 3.14159265358979);
        if (steps % 100 == 99) {   // every two simulated seconds: where it is, which way it faces, how fast it goes
            const glm::dmat3 R = _duck.trunkRot();
            ofLogNotice("microduck") << "  t=" << (steps + 1) * 0.02 << " s  position (" << _duck.trunkPos().x << ", " << _duck.trunkPos().y
                                     << ")  heading " << std::atan2(R[0][1], R[0][0]) * 180.0 / 3.14159265358979 << " deg  speed in its own forward "
                                     << _duck.velocityBody().x << " m/s";
        }
        if (_duck.faulted() || _duck.uprightness() < 0.5 || _duck.trunkPos().z < 0.05) break;
    }
    const double forward = glm::dot(_duck.trunkPos() - start, heading);
    const bool up = steps == total;
    const bool ok = up && forward > 0.5 * 0.3 * _checkSeconds && worstFk < 0.01 && !_duck.faulted();
    ofLogNotice("microduck") << (ok ? "PASS" : "FAIL") << ": " << (up ? "stayed up" : "FELL") << " for " << steps * 0.02
                             << " s, walked " << forward << " m forward, heading changed up to " << worstDrift << " deg, worst tilt (up = 1) " << worstUp
                             << ", ofxRobot FK vs simulation: worst " << worstFk << " mm, policy "
                             << (steps ? policyUs / steps : 0.0) << " us/step, safety gate "
                             << (_duck.faulted() ? "FAULT: " + _duck.faultReason() : std::string("ok"));
    for (const auto& l : _duck.limbs())
        ofLogNotice("microduck") << "  " << l.name << ": FK vs simulation " << fkAtRest[l.name] << " mm at rest, worst " << fkWorst[l.name] << " mm walking";
    ofExit(ok ? 0 : 1);
}

void ofApp::update() {
    if (_check) return;
    std::lock_guard<std::mutex> lock(_duck.mutex);   // the simulation runs on its own thread
    const double wall = ofGetLastFrameTime();
    _duck.setCommand(_vx, _vy, _wz);
    const double sim = _duck.simTime();
    if (wall > 0.0 && sim >= _lastSim) _realtime += 0.1 * ((sim - _lastSim) / wall - _realtime);
    _lastSim = sim;

    const glm::vec3 p(_duck.trunkPos());
    _focus += (p - _focus) * 0.15f;
    _cam.setTarget(_focus);                       // re-aims with a Y-up default, so put Z back up
    _cam.lookAt(_focus, glm::vec3(0, 0, 1));
    if (_trail.size() == 0 || glm::distance(_trail.getVertices().back(), glm::vec3(p.x, p.y, 0.002f)) > 0.01f)
        _trail.addVertex(p.x, p.y, 0.002f);
    if (_trail.size() > 2000) _trail.getVertices().erase(_trail.getVertices().begin(), _trail.getVertices().begin() + 500);
}

glm::mat4 ofApp::toMat4(const double* pos, const double* mat) const {
    glm::mat4 M(1.f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) M[c][r] = (float)mat[3 * r + c];   // MuJoCo is row-major, glm column-major
    M[3] = glm::vec4((float)pos[0], (float)pos[1], (float)pos[2], 1.f);
    return M;
}

void ofApp::drawRobot() {
    const mjModel* m = _duck.model();
    const mjData* d = _duck.data();
    ofMaterial mat;
    mat.setShininess(30.f);
    for (int g = 0; g < m->ngeom; ++g) {
        if (m->geom_type[g] != mjGEOM_MESH) continue;
        const bool collision = m->geom_contype[g] != 0 || m->geom_conaffinity[g] != 0;
        if (collision != _showCollision) continue;     // draw the visual geometry, or (g) the collision geometry
        const float* c = m->geom_rgba + 4 * g;
        mat.setDiffuseColor(ofFloatColor(c[0], c[1], c[2], 1.f));
        mat.setSpecularColor(ofFloatColor(0.3f));
        mat.begin();
        ofPushMatrix();
        ofMultMatrix(toMat4(d->geom_xpos + 3 * g, d->geom_xmat + 9 * g));
        _meshes[m->geom_dataid[g]].draw();
        ofPopMatrix();
        mat.end();
    }
}

// ofxRobot's view of the robot: each limb's joints from RobotKinematics + ArmIk forward kinematics, and where that
// puts the foot, next to where the simulation has it.
void ofApp::drawKinematics() {
    glm::mat4 T(1.f);
    const glm::dmat3 R = _duck.trunkRot();
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) T[c][r] = (float)R[c][r];
    T[3] = glm::vec4(glm::vec3(_duck.trunkPos()), 1.f);
    ofDisableDepthTest();
    ofPushMatrix();
    ofMultMatrix(T);
    ofSetLineWidth(3.f);
    for (const Microduck::Limb& l : _duck.limbs()) {
        const std::vector<glm::dmat4> frames = l.ik.linkFrames(l.q);
        glm::vec3 prev(0.f);
        ofSetColor(80, 220, 255);
        for (const glm::dmat4& f : frames) {
            const glm::vec3 p(f[3]);
            ofDrawLine(prev, p);
            ofDrawSphere(p, 0.0035f);
            prev = p;
        }
        const glm::vec3 tool(_duck.toolFk(l));
        ofDrawLine(prev, tool);
        ofSetColor(255, 220, 60);
        ofDrawSphere(tool, 0.006f);                          // ofxRobot: the tool by forward kinematics
        ofSetColor(255, 80, 80, 200);
        ofDrawSphere(glm::vec3(_duck.toolSim(l)), 0.0035f);  // MuJoCo: the same body, from the simulation
    }
    ofSetLineWidth(1.f);
    ofPopMatrix();
    ofEnableDepthTest();
}

void ofApp::draw() {
    if (_check) return;
    std::lock_guard<std::mutex> lock(_duck.mutex);
    ofBackgroundGradient(ofColor(70, 78, 92), ofColor(18, 20, 26), OF_GRADIENT_LINEAR);
    ofEnableDepthTest();
    _cam.begin();

    // The floor: a plane with a grid on it that moves with the robot's footprint so the walking reads.
    ofSetColor(40);
    ofDrawBox(glm::vec3(_focus.x, _focus.y, -0.01f), 6.f, 6.f, 0.02f);
    ofSetColor(255, 255, 255, 50);
    ofPushMatrix();
    ofTranslate(std::round(_focus.x / 0.1f) * 0.1f, std::round(_focus.y / 0.1f) * 0.1f, 0.0005f);
    ofRotateYDeg(90);
    ofDrawGridPlane(0.1f, 30);
    ofPopMatrix();
    ofSetColor(255, 200, 80);
    ofSetLineWidth(2.f);
    _trail.draw();
    ofSetLineWidth(1.f);

    ofEnableLighting();
    _light.enable();
    drawRobot();
    _light.disable();
    ofDisableLighting();
    {   // the look-at target: it moves with the robot
        const glm::dvec3 t = _duck.lookTarget();
        ofSetColor(255, 50, 50);
        ofDrawSphere(glm::vec3(t), 0.02f);
    }
    if (_showKinematics) drawKinematics();
    _cam.end();
    ofDisableDepthTest();

    const glm::dvec3 vb = _duck.velocityBody();
    const double tilt = std::acos(std::min(1.0, std::max(-1.0, _duck.uprightness()))) * 180.0 / 3.14159265358979;
    ofSetColor(255);
    std::string s = "microduck: MuJoCo physics, ofxOnnx policy, ofxRobot limbs\n";
    s += "command  forward " + ofToString(_duck.command().x, 2) + " m/s   sideways " + ofToString(_duck.command().y, 2) +
         " m/s   turn " + ofToString(_duck.command().z, 2) + " rad/s (given to the policy: " + ofToString(_duck.appliedCommand().z, 2) + ", heading held)\n";
    s += "walking  forward " + ofToString(vb.x, 2) + " m/s   sideways " + ofToString(vb.y, 2) + " m/s   tilt " +
         ofToString(tilt, 1) + " deg   sim " + ofToString(_duck.simTime(), 1) + " s (" + ofToString(_realtime, 2) + "x real time)\n";
    s += "policy " + ofToString(_duck.policyMicros(), 0) + " us/step   ofxRobot FK vs simulation: " +
         ofToString(_duck.maxFkErrorMm(), 2) + " mm   safety gate: " +
         (_duck.faulted() ? "FAULT " + _duck.faultReason() + " (f clears)" : std::string("ok")) + "\n";
    s += "up/down walk, left/right turn, , . sideways, space stop, x shove, r reset, p pause, k kinematics, g collision, f clear fault";
    ofDrawBitmapString(s, 12, 20);
}

void ofApp::keyPressed(int key) {
    std::lock_guard<std::mutex> lock(_duck.mutex);
    switch (key) {
        case OF_KEY_UP:    _vx += 0.1f; break;
        case OF_KEY_DOWN:  _vx -= 0.1f; break;
        case OF_KEY_LEFT:  _wz += 0.25f; break;
        case OF_KEY_RIGHT: _wz -= 0.25f; break;
        case ',': _vy += 0.05f; break;
        case '.': _vy -= 0.05f; break;
        case ' ': _vx = _vy = _wz = 0.f; break;
        case 'x': _duck.push(0.f, 0.25f); break;
        case 'r': _duck.reset(); _trail.clear(); break;
        case 'p': _duck.paused = !_duck.paused; break;
        case 'k': _showKinematics = !_showKinematics; break;
        case 'g': _showCollision = !_showCollision; break;
        case 'f': _duck.clearFault(); break;
        default: break;
    }
    _vx = glm::clamp(_vx, -0.2f, 0.4f);
    _vy = glm::clamp(_vy, -0.2f, 0.2f);
    _wz = glm::clamp(_wz, -1.f, 1.f);
}
