#pragma once
#include "ofMain.h"
#include "Swarm.h"

// Thirty microducks walking around a 4 m square arena, among obstacles, all in one MuJoCo world, all running the same trained policy (the
// NavigateWalkEnclosed one), each seeing the others and the walls with its time-of-flight sensor. See Swarm.h.
//
//   r                 reset: scatter them again
//   p                 pause
//   g                 show the collision shapes
//   drag, wheel       orbit / zoom the camera
//
// Command line (with --swarm): --check [seconds] runs the swarm headless for that long and exits 0 if every duck stayed up,
// printing how often ducks touched; --simplify draws reduced meshes.
class SwarmApp : public ofBaseApp {
public:
    SwarmApp(bool check = false, double checkSeconds = 30.0, bool simplify = false)
        : _check(check), _checkSeconds(checkSeconds), _simplify(simplify) {}

    void setup() override;
    void update() override;
    void draw() override;
    void exit() override;
    void keyPressed(int key) override;

private:
    void runCheck();
    void buildMeshes();

    bool _check, _simplify;
    double _checkSeconds;
    Swarm _swarm;
    std::vector<ofVboMesh> _meshes;                         // one per mesh of the model, in its geom's frame
    std::vector<std::vector<int>> _geomsOfMesh[2];          // [collision?][mesh] -> the geoms that use it, one or more per duck
    ofShader _shader;
    ofEasyCam _cam;
    bool _showCollision = false;
    float _half = 2.f;                                      // half the arena's side
};
