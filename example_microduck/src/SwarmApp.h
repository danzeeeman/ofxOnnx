#pragma once
#include "ofMain.h"
#include "Swarm.h"
#include "CameraPath.h"       // ofxRobot/src/core
#include "FrameRecorder.h"
#include "DomeLight.h"

// Sixteen microducks walking around a 4 m square arena, among obstacles, all in one MuJoCo world, all running the same trained policy (the
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

    // --render [dir] [--fps n] [--size WxH] [--seconds s]: write a 30 second camera flight over the arena as PNG frames,
    // frame by frame, and exit (ofxRobot's FrameRecorder.h; 1920x1080, 50 fps by default: one frame per policy period).
    // Nothing runs in real time: every frame the swarm takes the same steps on this thread, so the video is the same however
    // long a frame takes to draw. Join with: ffmpeg -framerate 50 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 14 ducks.mp4
    FrameRecorder::Options render;
    bool track = false;             // --track (with --lock): the camera rides behind one duck instead of looking down on the arena

private:
    void runCheck();
    void buildMeshes();
    void drawScene();
    void buildFlight();

    bool _check, _simplify;
    double _checkSeconds;
    Swarm _swarm;
    std::vector<ofVboMesh> _meshes;                         // one per mesh of the model, in its geom's frame
    std::vector<std::vector<int>> _geomsOfMesh[2];          // [collision?][mesh] -> the geoms that use it, one or more per duck
    ofShader _shader;
    ofEasyCam _cam;
    glm::vec3 _trackFocus{0.f};
    float _trackAz = 0.f;
    bool _trackInit = false;
    ofCamera _flightCam;            // the camera of a recording: placed by the flight, not the mouse
    DomeLight _dome;                // lights the walls and obstacles; the ducks are lit by the same dome in their shader
    FrameRecorder _rec;
    CameraPath _flight;
    double _flightT = 0.0, _simCarry = 0.0;
    bool _showCollision = false;
    float _half = 2.f;                                      // half the arena's side
};
