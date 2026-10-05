#pragma once
#include "ofMain.h"
#include "Microduck.h"

// A trained walking policy making a biped walk: the microduck, in MuJoCo, driven by ofxOnnx, described and checked
// with ofxRobot. See Microduck.h for who does what.
//
//   up / down         walk forward / back (velocity command, m/s)
//   left / right      turn
//   , and .           step sideways
//   space             stop (zero command)
//   x                 shove it
//   r                 reset
//   p                 pause
//   k                 show ofxRobot's kinematics: each limb's joints, and the foot where its forward kinematics puts it
//   g                 show the collision geometry
//   f                 clear a safety-gate fault
//   drag, wheel       orbit / zoom the camera (it follows the robot)
//
// Command line: --check [seconds] walks forward for that long without drawing anything and exits 0 or 1.
class ofApp : public ofBaseApp {
public:
    ofApp(bool check = false, double checkSeconds = 10.0) : _check(check), _checkSeconds(checkSeconds) {}

    void setup() override;
    void update() override;
    void draw() override;
    void keyPressed(int key) override;

private:
    void runCheck();
    void drawRobot();
    void drawKinematics();
    glm::mat4 toMat4(const double* pos, const double* mat) const;

    bool _check = false;
    double _checkSeconds = 10.0;

    Microduck _duck;
    std::vector<ofVboMesh> _meshes;      // every mesh in the model, in the model's own frame
    ofEasyCam _cam;
    ofLight _light;
    ofPolyline _trail;
    glm::vec3 _focus{0.f};

    float _vx = 0.2f, _vy = 0.f, _wz = 0.5f;   // it starts out walking a circle (0.4 m radius)
    bool _showKinematics = true, _showCollision = false;
    double _realtime = 1.0;
    double _lastWall = 0.0, _lastSim = 0.0;
};
