#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"

// The smallest ofxOnnx example: load a model, run it, check the answer, time it.
//
// bin/data/mlp_demo.onnx is a tiny MLP (16 inputs -> 32 ELU units -> 8 outputs).
// bin/data/mlp_demo.expected.json holds an input and the output Python's onnxruntime gives for it, so
// this checks that what comes out of ofxOnnx is the same number for number, on the platform it runs on.
//
// Both ways of running a model are shown:
//   runFloat(in, out)   one float vector in, one out: the easy way, for a policy or regressor in a loop
//   run(tensor)         ONNX Runtime's own tensors, for models with several inputs/outputs or other types
//
// Run with --check to print PASS/FAIL and exit 0/1.
class ofApp : public ofBaseApp {
public:
    explicit ofApp(bool check = false) : _check(check) {}

    void setup() override;
    void draw() override;

private:
    bool _check = false;
    ofxOnnx _onnx;

    std::vector<float> _input, _expected, _got;
    double _maxErrFloat = 0.0, _maxErrTensor = 0.0, _microsPerRun = 0.0;
    int _runs = 0;
    bool _ok = false;
    std::string _problem;
};
