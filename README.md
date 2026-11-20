# Onnx Runtime
Wrapper for the Microsoft Onnx Runtime [onnx runtime](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages)

## Description
Perform inference or training using onnx models. 
CPU onnx libs are included for macOS, Linux and Windows (x64). 

## Windows
ONNX Runtime 1.24.4 (CPU) for Windows x64 is in `libs/onnxruntime/lib/vs/x64`. The project generator links `onnxruntime.lib` and
copies `onnxruntime.dll` and `onnxruntime_providers_shared.dll` next to the exe; a hand-written project does the same with
`AdditionalDependencies` and a post-build `copy` (see `example_inference/example_inference.vcxproj`).

Model paths are native strings internally (wide characters on Windows), so paths with spaces or non-ASCII characters work.
A model whose weights are in a companion `.onnx.data` file (PyTorch's exporter makes these) loads as long as the two files
stay in the same folder.

For GPU on Windows, use Microsoft's `onnxruntime-win-x64-gpu` package of the same version in place of these libraries
and define `OFX_ONNX_USE_CUDA`.

## Running a model
```cpp
ofxOnnx onnx;
ofxOnnx::Settings settings;
settings.modelPath = "policy.onnx";      // relative to bin/data
onnx.load(settings);

std::vector<float> in(127), out;         // fill `in` every frame
onnx.runFloat(in, out);                  // one float vector in, one out: policies, regressors, classifiers
```
`run(...)` and the tensor helpers are there for models with several inputs or outputs, or other types.

## Examples
`example_inference` loads a small demo model, checks that its output matches Python's onnxruntime number for number,
and times it. `example_inference.exe --check` prints PASS/FAIL and exits 0/1.

The robot examples that use ofxOnnx live in ofxRobot (`addons/ofxRobot`), which they build against with this addon next to it:
`example_microduck` (a biped walking on a trained policy, a chain of followers, or a swarm) and `example_quadruped` (an ANYmal-D on
Isaac Lab's pretrained policy and an LSTM actuator network, in ofxRobot's streaming-client shape). See their READMEs there.

`example_phi4` turns a garment description into a GarmentCode design (`design.json`) with the garmentgen model, a Phi-4-multimodal
fine-tune exported to ONNX (`onnx_export` in the garmentgen project; the folder with `phi-4-mm-text.onnx`, `phi-4-mm-embedding.onnx` and the
tokenizer goes in `example_phi4/bin/data/onnx`, or pass `--model <dir>`). ofxOnnx has no onnxruntime-genai, so the example carries its own
byte-level BPE tokenizer (`Tokenizer.cpp`) and a KV-cache decode loop (`Phi4Chat.cpp`) whose cache stays on the GPU through `Ort::IoBinding`.
It needs `onnxruntime_providers_cuda.dll` (`scripts/get_cuda_provider.ps1`, copied next to the exe), the CUDA 13 toolkit on the PATH and
cuDNN 9 for CUDA 13 (see Windows above); without cuDNN it runs on the CPU, about 50 times slower.
`example_phi4.exe --prompt "a knee-length sleeveless dress" --out design.json` writes one design and exits; `--cudnn <dir>` says where cuDNN is.

## Linux
### CUDA
CUDA tries to link by default. As outlined via the addon_config.mk file. 
Will fallback to CPU if not present. 

[Download the GPU libs](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4) for Linux. Unzip and put libs into `ofxOnnx / libs / onnxruntime / lib / linux64`

## macOS
### Copy dylibs in Xcode 
If there is an error regarding the onnx runtime dylib not linking.
Copy to the executables in the Build Phases tab.
<img width="1060" height="630" alt="image" src="https://github.com/user-attachments/assets/61cb70d5-a4af-4a3d-83c0-daf10527566f" />

