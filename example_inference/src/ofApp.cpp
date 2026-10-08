#include "ofApp.h"
#include <chrono>

void ofApp::setup() {
    ofSetFrameRate(60);

    // 1. Load the model (a path relative to bin/data). Settings has the thread count, CUDA / CoreML switches, etc.
    ofxOnnx::Settings settings;
    settings.modelPath = "mlp_demo.onnx";
    settings.bPrintModelInfo = true;   // prints the inputs and outputs the model declares
    settings.logLevel = OF_LOG_WARNING;   // ONNX Runtime's own log: warnings and errors only
    if (!_onnx.load(settings)) { _problem = "could not load the model"; if (_check) ofExit(1); return; }

    // 2. The reference: an input, and what Python's onnxruntime says the output is.
    const ofJson ref = ofLoadJson("mlp_demo.expected.json");
    _input = ref["input"].get<std::vector<float>>();
    _expected = ref["output"].get<std::vector<float>>();

    // 3. Run it with runFloat(): the output vector is filled in.
    if (!_onnx.runFloat(_input, _got)) { _problem = "runFloat failed"; if (_check) ofExit(1); return; }
    for (size_t i = 0; i < _expected.size() && i < _got.size(); ++i)
        _maxErrFloat = std::max(_maxErrFloat, (double)std::fabs(_got[i] - _expected[i]));

    // 4. And with ONNX Runtime's tensors, the general way: run() returns Ort::Values.
    std::vector<float> in = _input;
    std::vector<Ort::Value> outs = _onnx.run(in);
    if (!outs.empty()) {
        const float* p = outs[0].GetTensorData<float>();
        for (size_t i = 0; i < _expected.size(); ++i) _maxErrTensor = std::max(_maxErrTensor, (double)std::fabs(p[i] - _expected[i]));
    }

    // 5. How fast: the cost of one policy step.
    const int n = 20000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) _onnx.runFloat(_input, _got);
    _microsPerRun = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
    _runs = n;

    _ok = _got.size() == _expected.size() && _maxErrFloat < 1e-5 && _maxErrTensor < 1e-5;
    ofLogNotice("ofxOnnx example") << (_ok ? "PASS" : "FAIL") << ": ONNX Runtime " << Ort::GetVersionString()
                                   << "; max difference from the Python reference " << _maxErrFloat << " (runFloat), "
                                   << _maxErrTensor << " (tensors); " << _microsPerRun << " us per run";
    if (_check) ofExit(_ok ? 0 : 1);
}

void ofApp::draw() {
    ofBackground(24);
    ofSetColor(255);
    std::string s = "ofxOnnx example   ONNX Runtime " + Ort::GetVersionString() + "\n";
    if (!_problem.empty()) s += "\nPROBLEM: " + _problem;
    else {
        s += "\nmlp_demo.onnx: 16 inputs -> 8 outputs\n\n";
        s += std::string(_ok ? "PASS" : "FAIL") + "  matches Python's onnxruntime\n";
        s += "  runFloat():   max difference " + ofToString(_maxErrFloat, 8) + "\n";
        s += "  run(tensor):  max difference " + ofToString(_maxErrTensor, 8) + "\n\n";
        s += ofToString(_microsPerRun, 1) + " microseconds per run (" + ofToString(_runs) + " runs)";
    }
    ofDrawBitmapString(s, 20, 30);

    // The eight outputs: the Python reference (grey) and what ofxOnnx returned (green).
    const float x0 = 60.f, y0 = 400.f, w = 40.f, scale = 40.f;
    ofSetColor(70);
    ofDrawLine(x0 - 10, y0, x0 + 8 * (w + 10), y0);
    for (size_t i = 0; i < _expected.size() && i < _got.size(); ++i) {
        const float x = x0 + i * (w + 10);
        ofSetColor(110);
        ofDrawRectangle(x, y0, w * 0.45f, -_expected[i] * scale);
        ofSetColor(80, 220, 120);
        ofDrawRectangle(x + w * 0.5f, y0, w * 0.45f, -_got[i] * scale);
        ofSetColor(200);
        ofDrawBitmapString(ofToString(i), x + 12, y0 + 60);
    }
}
