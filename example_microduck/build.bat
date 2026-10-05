@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\src\openFrameworks\addons\ofxOnnx\example_microduck
msbuild example_microduck.vcxproj /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
