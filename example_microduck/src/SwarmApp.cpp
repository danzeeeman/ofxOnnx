#include "SwarmApp.h"
#include <unordered_map>

namespace {
const char* kVert = R"(#version 150
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
const char* kFrag = R"(#version 150
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
    // the ducks' own colours are dark (black plastic, orange trim), so the dome is strong: about twice what a pale surface would take
    vec3 base = max(vColor, vec3(0.16));
    outColor = vec4(base * (1.7 * dome + 0.6 * diff) + vec3(0.35 * spec), 1.0);
})";
}

void SwarmApp::setup() {
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    if (!_swarm.setup("microduck")) { ofLogError("SwarmApp") << "setup failed"; ofExit(1); return; }
    if (_check) { runCheck(); return; }

    _shader.setupShaderFromSource(GL_VERTEX_SHADER, kVert);
    _shader.setupShaderFromSource(GL_FRAGMENT_SHADER, kFrag);
    _shader.bindDefaults();
    _shader.linkProgram();
    buildMeshes();

    _cam.setUpAxis(glm::vec3(0, 0, 1));
    _cam.setNearClip(0.01f);
    _cam.setFarClip(100.f);
    const int wall0 = mj_name2id(_swarm.model(), mjOBJ_GEOM, "wall0");
    _half = wall0 >= 0 ? (float)_swarm.model()->geom_pos[3 * wall0 + 1] - 0.05f : 2.f;        // the arena's half side, from its wall
    _cam.setTarget(glm::vec3(0.f, 0.f, 0.1f));
    _cam.setPosition(glm::vec3(1.3f, -2.1f, 1.7f) * _half);
    _cam.lookAt(glm::vec3(0.f, 0.f, 0.1f), glm::vec3(0, 0, 1));
    _dome.setup();
    if (render.enabled()) {
        buildFlight();
        _flightCam.setNearClip(0.01f);
        _flightCam.setFarClip(100.f);
        _rec.setup(render, _flight.length());
        _flightT = -render.warmup;
    } else {
        _swarm.startThread();
    }
}

void SwarmApp::exit() {
    _rec.finish();
    _swarm.stopThread();
}

// The flight for a video, 30 seconds over the arena: high and wide, round to the side, low across the floor, in close
// behind one duck as it walks (the camera follows it), and back out. The azimuth gains a turn over the loop.
void SwarmApp::buildFlight() {
    const glm::vec3 mid(0.f, 0.f, 0.12f);
    const float H = _half;
    _flight.shots = {
        {mid, -60.f, 34.f, 1.7f * H, 6.f},     // the arena from a corner, in tight
        {mid, 20.f, 28.f, 1.5f * H, 6.f},      // round to the side
        {mid, 100.f, 11.f, 1.5f * H, 6.f},     // low across the floor
        {mid, 190.f, 20.f, 1.1f * H, 6.f},     // (the follow shot picks it up here)
        {mid, 270.f, 32.f, 1.8f * H, 6.f},     // and back out
    };
    _flight.azimuthPerLoop = 360.f;
}

// The model's meshes, as MuJoCo compiled them (each already sits in its geom's frame), as a triangle soup with flat normals; with
// --simplify, merged on a 2 mm grid with smooth normals instead (~45k triangles in all rather than ~800k per duck).
void SwarmApp::buildMeshes() {
    const mjModel* m = _swarm.model();
    for (int i = 0; i < m->nmesh; ++i) {
        ofVboMesh mesh;
        mesh.setMode(OF_PRIMITIVE_TRIANGLES);
        const int v0 = m->mesh_vertadr[i], f0 = m->mesh_faceadr[i], nv = m->mesh_vertnum[i];
        auto vert = [&](int v) { return glm::vec3(m->mesh_vert[3 * (v0 + v)], m->mesh_vert[3 * (v0 + v) + 1], m->mesh_vert[3 * (v0 + v) + 2]); };
        if (!_simplify) {
            for (int f = 0; f < m->mesh_facenum[i]; ++f) {
                const glm::vec3 p[3] = {vert(m->mesh_face[3 * (f0 + f)]), vert(m->mesh_face[3 * (f0 + f) + 1]), vert(m->mesh_face[3 * (f0 + f) + 2])};
                glm::vec3 n = glm::cross(p[1] - p[0], p[2] - p[0]);
                n = glm::length(n) > 0.f ? glm::normalize(n) : glm::vec3(0, 0, 1);
                for (int c = 0; c < 3; ++c) { mesh.addVertex(p[c]); mesh.addNormal(n); }
            }
        } else {
            const float cell = 0.002f;
            std::unordered_map<long long, int> cellOf;
            std::vector<int> clusterOf(nv);
            std::vector<glm::vec3> sum, normal;
            std::vector<int> count;
            for (int v = 0; v < nv; ++v) {
                const glm::vec3 p = vert(v);
                const long long kx = (long long)std::floor(p.x / cell) + 100000, ky = (long long)std::floor(p.y / cell) + 100000, kz = (long long)std::floor(p.z / cell) + 100000;
                const long long key = (kx << 42) ^ (ky << 21) ^ kz;
                auto it = cellOf.find(key);
                if (it == cellOf.end()) { it = cellOf.emplace(key, (int)sum.size()).first; sum.emplace_back(0.f); normal.emplace_back(0.f); count.push_back(0); }
                clusterOf[v] = it->second; sum[it->second] += p; count[it->second]++;
            }
            for (int f = 0; f < m->mesh_facenum[i]; ++f) {
                const int a = clusterOf[m->mesh_face[3 * (f0 + f)]], b = clusterOf[m->mesh_face[3 * (f0 + f) + 1]], c = clusterOf[m->mesh_face[3 * (f0 + f) + 2]];
                if (a == b || b == c || a == c) continue;
                const glm::vec3 pa = sum[a] / float(count[a]), pb = sum[b] / float(count[b]), pc = sum[c] / float(count[c]);
                const glm::vec3 n = glm::cross(pb - pa, pc - pa);
                normal[a] += n; normal[b] += n; normal[c] += n;
                mesh.addIndex(a); mesh.addIndex(b); mesh.addIndex(c);
            }
            for (size_t k = 0; k < sum.size(); ++k) {
                mesh.addVertex(sum[k] / float(count[k]));
                mesh.addNormal(glm::length(normal[k]) > 0.f ? glm::normalize(normal[k]) : glm::vec3(0, 0, 1));
            }
        }
        _meshes.push_back(mesh);
    }
    for (auto& v : _geomsOfMesh) v.assign(m->nmesh, {});
    for (int g = 0; g < m->ngeom; ++g) {
        if (m->geom_type[g] != mjGEOM_MESH) continue;
        const bool collision = m->geom_contype[g] != 0 || m->geom_conaffinity[g] != 0;
        _geomsOfMesh[collision][m->geom_dataid[g]].push_back(g);
    }
}

void SwarmApp::update() {
    if (!_rec.active()) return;
    // Fixed steps: the swarm's policy period is 20 ms; a frame is 1/fps, so the policy steps owed per frame are carried over.
    _simCarry += 0.02 > 0.0 ? (1.0 / _rec.options().fps) / 0.02 : 0.0;
    while (_simCarry >= 1.0) {
        _swarm.policyStep();
        _simCarry -= 1.0;
    }
    _flightT += _rec.dt();
}

void SwarmApp::draw() {
    if (_check) return;
    if (_rec.active()) {
        if (_rec.beginFrame()) { drawScene(); _rec.endFrame(); }
        return;
    }
    drawScene();
}

void SwarmApp::drawScene() {
    ofBackgroundGradient(ofColor(70, 78, 92), ofColor(18, 20, 26), OF_GRADIENT_LINEAR);
    ofEnableDepthTest();
    ofCamera* cam = &_cam;
    if (_rec.active()) {   // the flight, with the camera drawn in behind one duck for the middle of it
        const bool locked = _rec.options().lockCamera >= 0.0;                                       // --lock: one fixed view
        CameraShot s = _flight.at(locked ? _rec.options().lockCamera : _flightT);
        const float t = (float)std::max(_flightT, 0.0), w0 = 15.f, w1 = 25.f;                      // the follow shot: seconds 15 to 25
        if (locked) s = {glm::vec3(0.f, 0.f, 0.f), -90.f, 89.f, 1.35f * _half, 6.f};              // straight down on the arena
        float w = locked ? 0.f : glm::clamp(std::min((t - w0) / 3.f, (w1 - t) / 3.f), 0.f, 1.f);
        w = w * w * (3.f - 2.f * w);
        if (w > 0.f) {
            const glm::dvec3 p = _swarm.pos(3 % std::max(1, _swarm.count()));
            const double h = _swarm.heading(3 % std::max(1, _swarm.count()));
            s.target = glm::mix(s.target, glm::vec3((float)p.x, (float)p.y, 0.1f), w);
            s.azimuthDeg = glm::mix(s.azimuthDeg, glm::degrees((float)h) + 180.f - 25.f, w);        // behind it, a little to the side
            s.elevationDeg = glm::mix(s.elevationDeg, 17.f, w);
            s.distance = glm::mix(s.distance, 1.1f, w);
        }
        if (locked && track) {   // a tracking shot: low in front of one duck, a little to the side, looking back at it, the camera easing after it
            const int k = 3 % std::max(1, _swarm.count());
            const glm::dvec3 p = _swarm.pos(k);
            const float h = glm::degrees((float)_swarm.heading(k)) + 25.f;
            if (!_trackInit) { _trackFocus = glm::vec3((float)p.x, (float)p.y, 0.1f); _trackAz = h; _trackInit = true; }
            _trackFocus += (glm::vec3((float)p.x, (float)p.y, 0.1f) - _trackFocus) * 0.15f;
            _trackAz += std::atan2(std::sin(glm::radians(h - _trackAz)), std::cos(glm::radians(h - _trackAz))) * 57.29578f * 0.05f;
            s = {_trackFocus, _trackAz, 38.f, 1.3f, 6.f};
        }
        const float az = glm::radians(s.azimuthDeg), el = glm::radians(s.elevationDeg);
        _flightCam.setPosition(s.target + glm::vec3(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)) * s.distance);
        _flightCam.lookAt(s.target, glm::vec3(0.f, 0.f, 1.f));
        cam = &_flightCam;
    }
    cam->begin();

    ofSetColor(40);
    ofDrawBox(glm::vec3(0.f, 0.f, -0.01f), 2.f * _half + 4.f, 2.f * _half + 4.f, 0.02f);
    ofSetColor(255, 255, 255, 50);
    ofPushMatrix();
    ofTranslate(0.f, 0.f, 0.0005f);
    ofRotateYDeg(90);
    ofDrawGridPlane(0.1f, (size_t)((_half + 2.f) / 0.1f));
    ofPopMatrix();

    const mjModel* m = _swarm.model();
    // the poses, copied under the lock so that the threads are not held up while drawing
    std::vector<glm::mat4> pose(m->ngeom);
    {
        std::lock_guard<std::mutex> lock(_swarm.mutex);
        const mjData* d = _swarm.data();
        for (int g = 0; g < m->ngeom; ++g) {
            if (m->geom_type[g] != mjGEOM_MESH) continue;
            glm::mat4 M(1.f);
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M[c][r] = (float)d->geom_xmat[9 * g + 3 * r + c];   // row-major to column-major
            M[3] = glm::vec4((float)d->geom_xpos[3 * g], (float)d->geom_xpos[3 * g + 1], (float)d->geom_xpos[3 * g + 2], 1.f);
            pose[g] = M;
        }
    }
    // the walls and the obstacles (box and cylinder geoms): in the colours the model gives them, lit by the dome light
    ofEnableLighting();
    _dome.enable();
    for (int g = 0; g < m->ngeom; ++g) {
        if (m->geom_type[g] != mjGEOM_BOX && m->geom_type[g] != mjGEOM_CYLINDER) continue;
        if (_rec.active()) {   // in a recording the arena's walls are not drawn (they still hold the ducks in)
            const char* name = mj_id2name(m, mjOBJ_GEOM, g);
            if (name && std::string(name).rfind("wall", 0) == 0) continue;
        }
        const ofFloatColor colour(m->geom_rgba[4 * g], m->geom_rgba[4 * g + 1], m->geom_rgba[4 * g + 2], 1.f);
        ofMaterial material;
        material.setDiffuseColor(colour);
        material.setAmbientColor(colour * 0.5f);
        material.setShininess(20.f);
        material.begin();
        ofPushMatrix();
        glm::mat4 M(1.f);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M[c][r] = (float)_swarm.data()->geom_xmat[9 * g + 3 * r + c];
        M[3] = glm::vec4((float)_swarm.data()->geom_xpos[3 * g], (float)_swarm.data()->geom_xpos[3 * g + 1], (float)_swarm.data()->geom_xpos[3 * g + 2], 1.f);
        ofMultMatrix(M);
        if (m->geom_type[g] == mjGEOM_BOX) {
            ofDrawBox(0.f, 0.f, 0.f, 2.f * (float)m->geom_size[3 * g], 2.f * (float)m->geom_size[3 * g + 1], 2.f * (float)m->geom_size[3 * g + 2]);
        } else {                                              // a post: MuJoCo's cylinder stands along z, openFrameworks' along y
            ofRotateXDeg(90.f);
            ofDrawCylinder(0.f, 0.f, 0.f, (float)m->geom_size[3 * g], 2.f * (float)m->geom_size[3 * g + 1]);
        }
        ofPopMatrix();
        material.end();
    }
    _dome.disable();
    ofDisableLighting();

    // the ducks: one instanced draw per mesh, the instances being the ducks' copies of it
    const glm::vec4 light = ofGetCurrentViewMatrix() * glm::vec4(1.0f, -1.5f, 3.0f, 1.f);
    _shader.begin();
    _shader.setUniform3f("lightPos", glm::vec3(light));
    _shader.setUniform3f("up", glm::vec3(ofGetCurrentViewMatrix() * glm::vec4(0.f, 0.f, 1.f, 0.f)));
    const auto& groups = _geomsOfMesh[_showCollision ? 1 : 0];
    for (size_t mesh = 0; mesh < groups.size(); ++mesh) {
        const std::vector<int>& geoms = groups[mesh];
        for (size_t first = 0; first < geoms.size(); first += 16) {
            const int n = (int)std::min<size_t>(16, geoms.size() - first);
            std::vector<glm::mat4> models(n);
            std::vector<glm::vec3> colors(n);
            for (int i = 0; i < n; ++i) {
                const int g = geoms[first + i];
                models[i] = pose[g];
                const float* c = m->geom_rgba + 4 * g;
                const int duck = _swarm.duckOfBody(m->geom_bodyid[g]);
                const ofFloatColor tint = ofFloatColor::fromHsb(std::fmod(duck * 0.618f, 1.f), 0.45f, 1.f);   // every duck its own hue
                colors[i] = glm::vec3(c[0] * tint.r, c[1] * tint.g, c[2] * tint.b);
            }
            _shader.setUniformMatrix4f("models", models[0], n);
            _shader.setUniform3fv("colors", &colors[0].x, n);
            _meshes[mesh].drawInstanced(OF_MESH_FILL, n);
        }
    }
    _shader.end();
    cam->end();
    ofDisableDepthTest();
    if (_rec.active()) return;   // no text in a recording

    Swarm::Stats st;
    double wall = 0.0;
    {
        std::lock_guard<std::mutex> lock(_swarm.mutex);
        st = _swarm.stats();
        wall = _swarm.simTime();
    }
    ofSetColor(255);
    std::string s = "microduck swarm: " + ofToString(_swarm.count()) + " ducks in one MuJoCo world, one ofxOnnx policy (NavigateWalkEnclosed), each duck sees the others with its time-of-flight sensor\n";
    s += "sim " + ofToString(wall, 1) + " s   " + ofToString(ofGetFrameRate(), 0) + " fps   policy " + ofToString(_swarm.policyMicros() / 1000.0, 2) + " ms for all " + ofToString(_swarm.count()) + "\n";
    int seeing = 0;
    { std::lock_guard<std::mutex> lock(_swarm.mutex); for (int k = 0; k < _swarm.count(); ++k) seeing += _swarm.raysOnDucks(k) > 0; }
    s += "ducks that see another duck: " + ofToString(seeing) + "   touches: duck-obstacle " + ofToString(st.obstacleContactPeriods) + " periods   duck-duck " + ofToString(st.duckContactPeriods) + " periods (" + ofToString(st.duckPairs) + " pairs), duck-wall " + ofToString(st.wallContactPeriods) +
         " periods   closest trunks " + ofToString(st.closest, 2) + " m\n";
    s += "r reset, p pause, g collision shapes, drag/wheel camera";
    ofDrawBitmapString(s, 12, 20);
}

void SwarmApp::keyPressed(int key) {
    std::lock_guard<std::mutex> lock(_swarm.mutex);
    switch (key) {
        case 'r': _swarm.reset(); break;
        case 'p': _swarm.paused = !_swarm.paused; break;
        case 'g': _showCollision = !_showCollision; break;
        default: break;
    }
}

// The swarm without a window: every duck walks for `seconds` of simulated time. Passes when none of them fell over; the
// contacts between ducks and with the walls are counted and reported (the policy steers clear, it is not a guarantee).
void SwarmApp::runCheck() {
    const int n = _swarm.count();
    const int total = (int)(_checkSeconds / 0.02);
    std::vector<glm::dvec2> start(n);
    for (int k = 0; k < n; ++k) start[k] = glm::dvec2(_swarm.pos(k));
    double travelled = 0.0;
    int steps = 0;
    const auto wallStart = std::chrono::steady_clock::now();
    for (; steps < total; ++steps) {
        _swarm.policyStep();
        if (steps % 100 == 99) {
            const Swarm::Stats st = _swarm.stats();
            int seeing = 0, seeingWalls = 0;
            for (int k = 0; k < n; ++k) { seeing += _swarm.raysOnDucks(k) > 0; seeingWalls += _swarm.raysOnWalls(k) > 0; }
            ofLogNotice("swarm") << "  ducks seeing another duck: " << seeing << ", seeing a wall: " << seeingWalls << " of " << n;
            ofLogNotice("swarm") << "  t=" << (steps + 1) * 0.02 << " s  duck-duck contact periods " << st.duckContactPeriods << " (" << st.duckPairs
                                 << " pairs), duck-wall " << st.wallContactPeriods << ", closest trunks " << st.closest << " m, policy " << _swarm.policyMicros() / 1000.0 << " ms";
        }
    }
    for (int k = 0; k < n; ++k) travelled += glm::length(glm::dvec2(_swarm.pos(k)) - start[k]);
    const double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    const Swarm::Stats st = _swarm.stats();
    const bool ok = steps == total;                           // it runs to the end; a duck that falls over is simply stood up again
    ofLogNotice("swarm") << (ok ? "PASS" : "FAIL") << ": " << n << " ducks, " << steps * 0.02 << " s, mean distance from start " << travelled / n << " m; duck-duck contact in " << st.duckContactPeriods << " periods (" << st.duckPairs
                         << " pairs of " << n * (n - 1) / 2 << "), duck-wall in " << st.wallContactPeriods << " periods, duck-obstacle in " << st.obstacleContactPeriods << " periods, closest trunks " << st.closest << " m";
    ofLogNotice("swarm") << "per 20 ms period: observations " << _swarm.microsObserve() / 1000.0 << " ms, policy " << _swarm.microsInfer() / 1000.0 << " ms, servos + physics " << _swarm.microsStep() / 1000.0 << " ms";
    ofLogNotice("swarm") << "simulated " << steps * 0.02 << " s in " << wallSeconds << " s of computing: " << steps * 0.02 / wallSeconds << "x real time (every duck on one thread)";
    ofExit(ok ? 0 : 1);
}
