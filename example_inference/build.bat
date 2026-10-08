@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\src\openFrameworks\addons\ofxOnnx\example_inference
msbuild example_inference.vcxproj /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
