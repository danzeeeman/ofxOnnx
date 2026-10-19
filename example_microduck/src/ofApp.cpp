#include "ofApp.h"
#include <unordered_map>

void ofApp::setup() {
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    ofSetSphereResolution(10);

    if (!_duck.setup("microduck")) { ofLogError("ofApp") << "setup failed"; ofExit(1); return; }
    if (_startRun) _duck.setSkill(Microduck::Run);
    _duck.setSpawn(0.0, 0.0, 0.0);
    _duck.reset();
    if (_chain) {   // the followers in a line behind it, each following the one ahead
        for (int i = 0; i < _followerCount; ++i) {
            auto f = std::make_unique<Microduck>();
            if (!f->setup("microduck", 1u << Microduck::Follow)) { ofLogError("ofApp") << "follower setup failed"; ofExit(1); return; }
            f->setSpawn(-0.8 * (i + 1), 0.0, 0.0);
            f->reset();
            f->follow(i == 0 ? &_duck : _followers.back().get());
            _followers.push_back(std::move(f));
        }
    }
    if (_check) { if (_chain && !_startRun) runChainCheck(); else runCheck(); return; }

    // The robots are drawn instanced: a mesh is drawn once for all of them, with each robot's transform and colour picked by
    // gl_InstanceID. A Blinn-Phong point light, in view space.
    const std::string vert = R"(#version 150
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 models[16];
uniform vec3 colors[16];
in vec4 position;
in vec3 normal;
out vec3 vNormal;
out vec3 vPos;
out vec3 vColor;
void main() {
    mat4 M = models[gl_InstanceID];
    vec4 viewPos = modelViewMatrix * (M * position);
    vPos = viewPos.xyz;
    vNormal = mat3(modelViewMatrix) * (mat3(M) * normal);
    vColor = colors[gl_InstanceID];
    gl_Position = projectionMatrix * viewPos;
})";
    const std::string frag = R"(#version 150
uniform vec3 lightPos;   // view space
uniform vec3 up;         // world up, in view space
in vec3 vNormal;
in vec3 vPos;
in vec3 vColor;
out vec4 outColor;
void main() {
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 l = normalize(lightPos - vPos);
    vec3 v = normalize(-vPos);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, normalize(l + v)), 0.0), 30.0);
    // a dome light: the sky overhead, the ground bounce below, whatever way the surface faces; and a key light for form
    float sky = 0.5 + 0.5 * dot(n, normalize(up));
    vec3 dome = mix(vec3(0.55, 0.52, 0.50), vec3(1.0, 1.0, 1.06), sky);
    // the robot's own colours are dark (black plastic, orange trim), so the dome is strong
    vec3 base = max(vColor, vec3(0.16));
    outColor = vec4(base * (1.7 * dome + 0.6 * diff) + vec3(0.35 * spec), 1.0);
})";
    _shader.setupShaderFromSource(GL_VERTEX_SHADER, vert);
    _shader.setupShaderFromSource(GL_FRAGMENT_SHADER, frag);
    _shader.bindDefaults();
    _shader.linkProgram();

    // The model's meshes, as MuJoCo compiled them (so each already sits in its geom's frame): the CAD meshes, ~800k triangles
    // per robot. With --simplify the vertices are merged on a 2 mm grid (vertex clustering) and given smooth normals instead:
    // ~45k triangles in all, for a GPU that cannot draw them all. The physics does not use these.
    const mjModel* m = _duck.model();
    size_t before = 0, after = 0;
    for (int i = 0; i < m->nmesh; ++i) {
        ofVboMesh mesh;
        mesh.setMode(OF_PRIMITIVE_TRIANGLES);
        const int v0 = m->mesh_vertadr[i], f0 = m->mesh_faceadr[i], nv = m->mesh_vertnum[i];
        if (!_simplify) {   // the CAD meshes as they are: a triangle soup with flat normals
            for (int f = 0; f < m->mesh_facenum[i]; ++f) {
                glm::vec3 p[3];
                for (int c = 0; c < 3; ++c) {
                    const int vi = m->mesh_face[3 * (f0 + f) + c];
                    p[c] = glm::vec3(m->mesh_vert[3 * (v0 + vi)], m->mesh_vert[3 * (v0 + vi) + 1], m->mesh_vert[3 * (v0 + vi) + 2]);
                }
                glm::vec3 n = glm::cross(p[1] - p[0], p[2] - p[0]);
                n = glm::length(n) > 0.f ? glm::normalize(n) : glm::vec3(0, 0, 1);
                for (int c = 0; c < 3; ++c) { mesh.addVertex(p[c]); mesh.addNormal(n); }
                ++before; ++after;
            }
            _meshes.push_back(mesh);
            continue;
        }
        const float cell = 0.002f;
        std::unordered_map<long long, int> cellOf;
        std::vector<int> clusterOf(nv);
        std::vector<glm::vec3> sum, normal;
        std::vector<int> count;
        for (int v = 0; v < nv; ++v) {
            const glm::vec3 p(m->mesh_vert[3 * (v0 + v)], m->mesh_vert[3 * (v0 + v) + 1], m->mesh_vert[3 * (v0 + v) + 2]);
            const long long kx = (long long)std::floor(p.x / cell) + 100000, ky = (long long)std::floor(p.y / cell) + 100000, kz = (long long)std::floor(p.z / cell) + 100000;
            const long long key = (kx << 42) ^ (ky << 21) ^ kz;
            auto it = cellOf.find(key);
            if (it == cellOf.end()) {
                it = cellOf.emplace(key, (int)sum.size()).first;
                sum.emplace_back(0.f); normal.emplace_back(0.f); count.push_back(0);
            }
            clusterOf[v] = it->second;
            sum[it->second] += p;
            count[it->second]++;
        }
        for (int f = 0; f < m->mesh_facenum[i]; ++f) {
            const int a = clusterOf[m->mesh_face[3 * (f0 + f)]], b = clusterOf[m->mesh_face[3 * (f0 + f) + 1]], c = clusterOf[m->mesh_face[3 * (f0 + f) + 2]];
            ++before;
            if (a == b || b == c || a == c) continue;            // collapsed
            const glm::vec3 pa = sum[a] / float(count[a]), pb = sum[b] / float(count[b]), pc = sum[c] / float(count[c]);
            const glm::vec3 n = glm::cross(pb - pa, pc - pa);    // (area weighted)
            normal[a] += n; normal[b] += n; normal[c] += n;
            mesh.addIndex(a); mesh.addIndex(b); mesh.addIndex(c);
            ++after;
        }
        for (size_t k = 0; k < sum.size(); ++k) {
            mesh.addVertex(sum[k] / float(count[k]));
            mesh.addNormal(glm::length(normal[k]) > 0.f ? glm::normalize(normal[k]) : glm::vec3(0, 0, 1));
        }
        _meshes.push_back(mesh);
    }
    ofLogNotice("ofApp") << "display meshes: " << (_simplify ? "simplified from " + ofToString(before) + " to " : "full, ") << after << " triangles";

    _light.setPointLight();
    _light.setPosition(0.6f, -0.8f, 1.6f);
    _light.setDiffuseColor(ofFloatColor(1.f));
    _light.setAmbientColor(ofFloatColor(0.4f));

    _cam.setUpAxis(glm::vec3(0, 0, 1));
    _cam.setNearClip(0.005f);
    _cam.setFarClip(50.f);
    _focus = glm::vec3(_duck.trunkPos());
    for (auto& f : _followers) _focus += glm::vec3(f->publishedPos());
    _focus /= float(1 + _followers.size());              // the middle of the group
    _cam.setTarget(_focus);
    _cam.setPosition(_focus + glm::vec3(0.35f, -0.45f, 0.22f) * (1.f + 0.9f * float(_followers.size())));   // further back for a longer chain
    _cam.lookAt(_focus, glm::vec3(0, 0, 1));
    if (render.enabled() && !_check) {
        const float H = 0.7f + 0.25f * float(_followers.size());   // the group is bigger with more followers
        _flight.shots = {
            {glm::vec3(0.f), -50.f, 22.f, 1.1f * H, 6.f},    // the group from a corner, in tight
            {glm::vec3(0.f), 20.f, 18.f, 0.8f * H, 6.f},     // round to the side
            {glm::vec3(0.f), 100.f, 10.f, 0.55f * H, 6.f},    // low, among them
            {glm::vec3(0.f), 200.f, 20.f, 0.85f * H, 6.f},    // from behind
            {glm::vec3(0.f), 270.f, 26.f, 1.15f * H, 6.f},    // and back out
        };
        _flight.azimuthPerLoop = 360.f;
        _rec.setup(render, _flight.length());
        _flightT = -render.warmup;
        _flightCam.setNearClip(0.005f);
        _flightCam.setFarClip(50.f);
        _vx = 0.25f; _vy = 0.f; _wz = 0.3f;                  // the walker walks a circle (about 0.8 m across the radius)
    } else if (!_check) {
        _duck.startThread();
        for (auto& f : _followers) f->startThread();
    }
}

// A walk without a window: forward at 0.3 m/s. Judged on staying up, getting somewhere, and ofxRobot's forward
// kinematics agreeing with the simulation.
void ofApp::runCheck() {
    _duck.reset();
    const bool run = _duck.skill() == Microduck::Run;
    if (run) _duck.placeTarget(3.0, 0.0);               // a target 3 m ahead: run to it and stop
    else _duck.setCommand(0.3f, 0.f, 0.f);
    const glm::dvec3 start = _duck.trunkPos();
    const double startDistance = _duck.targetDistance();
    const glm::dmat3 R0 = _duck.trunkRot();
    const glm::dvec3 heading = glm::normalize(glm::dvec3(R0[0][0], R0[0][1], 0.0));
    const double yaw0 = _duck.heading();
    double worstUp = 1.0, worstFk = 0.0, policyUs = 0.0, worstDrift = 0.0;
    std::map<std::string, double> fkAtRest, fkWorst;
    for (const auto& l : _duck.limbs()) fkAtRest[l.name] = glm::length(_duck.toolFk(l) - _duck.toolSim(l)) * 1000.0;
    int steps = 0;
    const int total = (int)(_checkSeconds / 0.02);
    if (_threaded) _duck.startThread();
    for (; steps < total; ++steps) {
        std::unique_lock<std::mutex> lock(_duck.mutex, std::defer_lock);
        if (_threaded) {   // real time: sample the robot every 20 ms of wall time
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            lock.lock();
        } else {
            _duck.policyStep();
        }
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
    if (_threaded) _duck.stopThread();
    const double forward = glm::dot(_duck.trunkPos() - start, heading);
    const bool up = steps == total;
    const bool ok = up && (run ? _duck.targetDistance() < 0.4 : forward > 0.5 * 0.3 * _checkSeconds) && worstFk < 0.01 && !_duck.faulted();
    if (run) ofLogNotice("microduck") << "run policy: started " << startDistance << " m from the target, ended " << _duck.targetDistance() << " m from it";
    ofLogNotice("microduck") << (ok ? "PASS" : "FAIL") << ": " << (up ? "stayed up" : "FELL") << " for " << steps * 0.02
                             << " s, walked " << forward << " m forward, heading changed up to " << worstDrift << " deg, worst tilt (up = 1) " << worstUp
                             << ", ofxRobot FK vs simulation: worst " << worstFk << " mm, policy "
                             << (steps ? policyUs / steps : 0.0) << " us/step, safety gate "
                             << (_duck.faulted() ? "FAULT: " + _duck.faultReason() : std::string("ok"));
    for (const auto& l : _duck.limbs())
        ofLogNotice("microduck") << "  " << l.name << ": FK vs simulation " << fkAtRest[l.name] << " mm at rest, worst " << fkWorst[l.name] << " mm walking";
    ofExit(ok ? 0 : 1);
}

void ofApp::exit() {
    _rec.finish();
    for (auto& f : _followers) f->stopThread();
    _duck.stopThread();
}

// The chain, without a window: the first robot walks forward at 0.3 m/s and each of the others follows the one ahead.
// Passes when all five stay up and every follower ends up within 1.2 m of the robot it follows (it was started 0.8 m behind
// it, and the policy keeps 0.6 m).
void ofApp::runChainCheck() {
    std::vector<Microduck*> bots{&_duck};
    for (auto& f : _followers) bots.push_back(f.get());
    for (Microduck* b : bots) b->reset();
    _duck.setCommand(0.3f, 0.f, 0.f);
    const int total = (int)(_checkSeconds / 0.02);
    std::vector<double> worstUp(bots.size(), 1.0), closest(bots.size(), 1e9);
    int steps = 0;
    for (; steps < total; ++steps) {
        for (Microduck* b : bots) b->policyStep();
        bool down = false;
        for (size_t i = 0; i < bots.size(); ++i) {
            worstUp[i] = std::min(worstUp[i], bots[i]->uprightness());
            if (i > 0) closest[i] = std::min(closest[i], glm::length(glm::dvec2(bots[i]->trunkPos()) - glm::dvec2(bots[i - 1]->trunkPos())));
            if (bots[i]->faulted() || bots[i]->uprightness() < 0.5) down = true;
        }
        if (steps % 150 == 149) {
            std::string line = "  t=" + ofToString((steps + 1) * 0.02, 0) + " s  x:";
            for (Microduck* b : bots) line += " " + ofToString(b->trunkPos().x, 2);
            line += "   gaps:";
            for (size_t i = 1; i < bots.size(); ++i) line += " " + ofToString(bots[i]->targetDistance(), 2);
            ofLogNotice("microduck") << line;
        }
        if (down) break;
    }
    bool ok = steps == total;
    for (size_t i = 0; i < bots.size(); ++i) {
        const double gap = i > 0 ? bots[i]->targetDistance() : 0.0;
        if (i > 0 && gap > 1.2) ok = false;
        ofLogNotice("microduck") << (i == 0 ? "  walker" : "  follower " + ofToString(i)) << ": worst tilt (up = 1) " << worstUp[i]
                                 << (i > 0 ? ", " + ofToString(gap, 2) + " m behind the one ahead at the end (closest " + ofToString(closest[i], 2) + " m)" : std::string(""))
                                 << (bots[i]->faulted() ? ", FAULT " + bots[i]->faultReason() : std::string(""));
    }
    ofLogNotice("microduck") << (ok ? "PASS" : "FAIL") << ": chain of " << bots.size() << ", " << steps * 0.02 << " s, the walker went "
                             << bots[0]->trunkPos().x << " m";
    ofExit(ok ? 0 : 1);
}

void ofApp::update() {
    if (_check) return;
    if (_rec.active()) {   // recording: the robots are stepped here, by the policy period, on this thread
        _duck.setCommand(_vx, _vy, _wz);
        _simCarry += (1.0 / _rec.options().fps) / 0.02;
        while (_simCarry >= 1.0) {
            _duck.policyStep();
            for (auto& f : _followers) f->policyStep();
            _simCarry -= 1.0;
        }
        glm::vec3 centre(_duck.trunkPos());
        for (auto& f : _followers) centre += glm::vec3(f->publishedPos());
        centre /= float(1 + _followers.size());
        _focus += (centre - _focus) * 0.2f;
        _flightT += _rec.dt();
        return;
    }
    std::lock_guard<std::mutex> lock(_duck.mutex);   // the simulation runs on its own thread
    const double wall = ofGetLastFrameTime();
    _duck.setCommand(_vx, _vy, _wz);
    if (_duck.skill() == Microduck::Run) {   // the arrow keys, held, move the target (relative to the way the robot faces)
        const double step = 0.8 * std::min(wall, 0.1);
        const double ahead = (ofGetKeyPressed(OF_KEY_UP) ? step : 0.0) - (ofGetKeyPressed(OF_KEY_DOWN) ? step : 0.0);
        const double left = (ofGetKeyPressed(OF_KEY_LEFT) ? step : 0.0) - (ofGetKeyPressed(OF_KEY_RIGHT) ? step : 0.0);
        if (ahead != 0.0 || left != 0.0) _duck.nudgeTarget(ahead, left);
    }
    const double sim = _duck.simTime();
    if (wall > 0.0 && sim >= _lastSim) _realtime += 0.1 * ((sim - _lastSim) / wall - _realtime);
    _lastSim = sim;

    const glm::vec3 p(_duck.trunkPos());
    glm::vec3 centre = p;                                        // the camera follows the whole group
    for (auto& f : _followers) centre += glm::vec3(f->publishedPos());
    centre /= float(1 + _followers.size());
    _focus += (centre - _focus) * 0.15f;
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

void ofApp::drawRobots() {
    // Every robot's geom poses, copied under that robot's lock (the leader's is held by draw()), so that the physics threads
    // are not held up while drawing.
    std::vector<Microduck*> bots{&_duck};
    for (auto& f : _followers) bots.push_back(f.get());
    const mjModel* m = _duck.model();                      // the same model, once per robot
    const int n = (int)bots.size();
    std::vector<std::vector<glm::mat4>> pose(n, std::vector<glm::mat4>(m->ngeom));
    for (int b = 0; b < n; ++b) {
        std::unique_lock<std::mutex> lock(bots[b]->mutex, std::defer_lock);
        if (b > 0) lock.lock();
        const mjData* d = bots[b]->data();
        for (int g = 0; g < m->ngeom; ++g)
            if (m->geom_type[g] == mjGEOM_MESH) pose[b][g] = toMat4(d->geom_xpos + 3 * g, d->geom_xmat + 9 * g);
    }
    static const glm::vec3 tints[5] = {glm::vec3(1.f), glm::vec3(1.f, 0.55f, 0.55f), glm::vec3(0.55f, 1.f, 0.55f), glm::vec3(0.55f, 0.7f, 1.f), glm::vec3(1.f, 1.f, 0.5f)};
    const glm::vec4 light = ofGetCurrentViewMatrix() * glm::vec4(0.6f, -0.8f, 1.6f, 1.f);
    _shader.begin();
    _shader.setUniform3f("lightPos", glm::vec3(light));
    _shader.setUniform3f("up", glm::vec3(ofGetCurrentViewMatrix() * glm::vec4(0.f, 0.f, 1.f, 0.f)));
    std::vector<glm::vec3> colors(n);
    for (int g = 0; g < m->ngeom; ++g) {
        if (m->geom_type[g] != mjGEOM_MESH) continue;
        const bool collision = m->geom_contype[g] != 0 || m->geom_conaffinity[g] != 0;
        if (collision != _showCollision) continue;         // draw the visual geometry, or (g) the collision geometry
        const float* c = m->geom_rgba + 4 * g;
        std::vector<glm::mat4> models(n);
        for (int b = 0; b < n; ++b) {
            models[b] = pose[b][g];
            colors[b] = glm::vec3(c[0], c[1], c[2]) * tints[b % 5];
        }
        _shader.setUniformMatrix4f("models", models[0], n);
        _shader.setUniform3fv("colors", &colors[0].x, n);
        _meshes[m->geom_dataid[g]].drawInstanced(OF_MESH_FILL, n);
    }
    _shader.end();
}

// ofxRobot's view of the robot: each limb's joints from RobotKinematics + ArmIk forward kinematics, and where that
// puts the foot, next to where the simulation has it.
void ofApp::drawKinematics(Microduck& d) {
    glm::mat4 T(1.f);
    const glm::dmat3 R = d.trunkRot();
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) T[c][r] = (float)R[c][r];
    T[3] = glm::vec4(glm::vec3(d.trunkPos()), 1.f);
    ofDisableDepthTest();
    ofPushMatrix();
    ofMultMatrix(T);
    ofSetLineWidth(3.f);
    for (const Microduck::Limb& l : d.limbs()) {
        const std::vector<glm::dmat4> frames = l.ik.linkFrames(l.q);
        glm::vec3 prev(0.f);
        ofSetColor(80, 220, 255);
        for (const glm::dmat4& f : frames) {
            const glm::vec3 p(f[3]);
            ofDrawLine(prev, p);
            ofDrawSphere(p, 0.0035f);
            prev = p;
        }
        const glm::vec3 tool(d.toolFk(l));
        ofDrawLine(prev, tool);
        ofSetColor(255, 220, 60);
        ofDrawSphere(tool, 0.006f);                          // ofxRobot: the tool by forward kinematics
        ofSetColor(255, 80, 80, 200);
        ofDrawSphere(glm::vec3(d.toolSim(l)), 0.0035f);  // MuJoCo: the same body, from the simulation
    }
    ofSetLineWidth(1.f);
    ofPopMatrix();
    ofEnableDepthTest();
}

void ofApp::draw() {
    if (_check) return;
    if (_rec.active()) {
        if (_rec.beginFrame()) { drawScene(); _rec.endFrame(); }
        return;
    }
    drawScene();
}

void ofApp::drawScene() {
    std::lock_guard<std::mutex> lock(_duck.mutex);
    ofBackgroundGradient(ofColor(70, 78, 92), ofColor(18, 20, 26), OF_GRADIENT_LINEAR);
    ofEnableDepthTest();
    ofCamera* cam = &_cam;
    if (_rec.active()) {   // the flight, round the middle of the group
        CameraShot s = _flight.at(_rec.options().lockCamera >= 0.0 ? _rec.options().lockCamera : _flightT);
        if (_rec.options().lockCamera >= 0.0) {   // --lock: the whole group in view as it walks its circle, not in tight on one robot
            const float H = 0.7f + 0.25f * float(_followers.size());
            s.distance = 2.1f * H / (float)_rec.options().zoom;
            s.elevationDeg = std::max(8.f, 30.f - (float)_rec.options().lowerDeg);
        }
        const float az = glm::radians(s.azimuthDeg), el = glm::radians(s.elevationDeg);
        const glm::vec3 target = _focus + glm::vec3(0.f, 0.f, 0.08f);
        _flightCam.setPosition(target + glm::vec3(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)) * s.distance);
        _flightCam.lookAt(target, glm::vec3(0.f, 0.f, 1.f));
        cam = &_flightCam;
    }
    cam->begin();

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

    drawRobots();
    if (_duck.skill() == Microduck::Walk) {   // the look-at target: it moves with the robot (the walk policy ignores it)
        const glm::dvec3 t = _duck.lookTarget();
        ofSetColor(255, 50, 50);
        ofDrawSphere(glm::vec3(t), 0.02f);
    } else {                                  // the run policy's target: a marker on the floor that stays where it is
        const glm::vec3 t(_duck.target());
        ofSetColor(255, 140, 20);
        ofDrawSphere(t, 0.03f);
        ofSetColor(255, 140, 20, 120);
        ofSetLineWidth(2.f);
        ofDrawLine(glm::vec3(t.x, t.y, 0.f), glm::vec3(t.x, t.y, 0.3f));
        ofNoFill();
        ofPushMatrix();
        ofTranslate(t.x, t.y, 0.003f);
        ofDrawCircle(0.f, 0.f, 0.15f);        // the radius within which it counts as arrived
        ofPopMatrix();
        ofFill();
        ofSetLineWidth(1.f);
    }
    if (_showKinematics) {                      // ofxRobot's view of the limbs: the leader's while driving, every robot's in a recording
        drawKinematics(_duck);
        if (_rec.active()) for (auto& f : _followers) drawKinematics(*f);
    }
    cam->end();
    ofDisableDepthTest();
    if (_rec.active()) return;   // no text in a recording

    const glm::dvec3 vb = _duck.velocityBody();
    const double tilt = std::acos(std::min(1.0, std::max(-1.0, _duck.uprightness()))) * 180.0 / 3.14159265358979;
    ofSetColor(255);
    const bool run = _duck.skill() == Microduck::Run;
    std::string s = std::string("microduck: MuJoCo physics, ofxOnnx policy, ofxRobot limbs     policy: ") + Microduck::skillName(_duck.skill()) +
                    (_duck.hasSkill(Microduck::Run) ? "  (1 walk, 2 run)" : "") + "\n";
    if (run) s += "target  " + ofToString(_duck.targetDistance(), 2) + " m away (hold the arrow keys to move it)\n";
    else s += "command  forward " + ofToString(_duck.command().x, 2) + " m/s   sideways " + ofToString(_duck.command().y, 2) +
         " m/s   turn " + ofToString(_duck.command().z, 2) + " rad/s (given to the policy: " + ofToString(_duck.appliedCommand().z, 2) + ", heading held)\n";
    s += "walking  forward " + ofToString(vb.x, 2) + " m/s   sideways " + ofToString(vb.y, 2) + " m/s   tilt " +
         ofToString(tilt, 1) + " deg   sim " + ofToString(_duck.simTime(), 1) + " s (" + ofToString(_realtime, 2) + "x real time)   " + ofToString(ofGetFrameRate(), 0) + " fps\n";
    if (!_followers.empty()) {
        s += "followers, each behind the one ahead (m): ";
        for (auto& f : _followers) s += ofToString(f->targetDistance(), 2) + (f->trackingLost() ? "(lost)  " : "  ");
        s += "\n";
    }
    s += "policy " + ofToString(_duck.policyMicros(), 0) + " us/step   ofxRobot FK vs simulation: " +
         ofToString(_duck.maxFkErrorMm(), 2) + " mm   safety gate: " +
         (_duck.faulted() ? "FAULT " + _duck.faultReason() + " (f clears)" : std::string("ok")) + "\n";
    s += run ? "up/down/left/right move the target, space target 2.5 m ahead, 1 walk, x shove, r reset, p pause, k kinematics, g collision, f clear fault"
             : "up/down walk, left/right turn, , . sideways, space stop, 2 run, x shove, r reset, p pause, k kinematics, g collision, f clear fault";
    ofDrawBitmapString(s, 12, 20);
}

void ofApp::keyPressed(int key) {
    std::lock_guard<std::mutex> lock(_duck.mutex);
    const bool walk = _duck.skill() == Microduck::Walk;
    switch (key) {
        case '1': _duck.setSkill(Microduck::Walk); break;
        case '2': _duck.setSkill(Microduck::Run); break;
        case OF_KEY_UP:    if (walk) _vx += 0.1f; break;      // (with the run policy the held arrows move the target: see update)
        case OF_KEY_DOWN:  if (walk) _vx -= 0.1f; break;
        case OF_KEY_LEFT:  if (walk) _wz += 0.25f; break;
        case OF_KEY_RIGHT: if (walk) _wz -= 0.25f; break;
        case ',': _vy += 0.05f; break;
        case '.': _vy -= 0.05f; break;
        case ' ': if (walk) _vx = _vy = _wz = 0.f; else _duck.placeTarget(2.5, 0.0); break;
        case 'x': _duck.push(0.f, 0.25f); break;
        case 'r':
            _duck.reset(); _trail.clear();
            for (auto& f : _followers) { std::lock_guard<std::mutex> fl(f->mutex); f->reset(); }
            break;
        case 'p': {
            const bool pause = !_duck.paused;
            _duck.paused = pause;
            for (auto& f : _followers) f->paused = pause;
            break;
        }
        case 'k': _showKinematics = !_showKinematics; break;
        case 'g': _showCollision = !_showCollision; break;
        case 'f': _duck.clearFault(); break;
        default: break;
    }
    _vx = glm::clamp(_vx, -0.2f, 0.4f);
    _vy = glm::clamp(_vy, -0.2f, 0.2f);
    _wz = glm::clamp(_wz, -1.f, 1.f);
}
