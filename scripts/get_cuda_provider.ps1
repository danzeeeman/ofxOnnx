# Windows: fetches onnxruntime's CUDA provider (onnxruntime_providers_cuda.dll, 211 MB - too big for git, and a public
# fork cannot take Git LFS uploads) from the same release as the rest of libs/onnxruntime/lib/vs/x64
# (onnxruntime-win-x64-gpu_cuda13-1.24.4) and puts it beside them. Without it everything runs on the CPU.
#
#   powershell -ExecutionPolicy Bypass -File scripts\get_cuda_provider.ps1
$ErrorActionPreference = "Stop"
$version = "1.24.4"
$name = "onnxruntime-win-x64-gpu_cuda13-$version"
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$version/$name.zip"
$dest = Join-Path $PSScriptRoot "..\libs\onnxruntime\lib\vs\x64"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "ofxOnnx_$name"

New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$zip = Join-Path $tmp "$name.zip"
Write-Host "downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
Expand-Archive -Path $zip -DestinationPath $tmp -Force
# the zip's folder is named for the GPU build without the cuda13 suffix
$lib = Get-ChildItem -Path $tmp -Directory | Where-Object { Test-Path (Join-Path $_.FullName "lib\onnxruntime_providers_cuda.dll") } | Select-Object -First 1
Copy-Item (Join-Path $lib.FullName "lib\onnxruntime_providers_cuda.dll") $dest -Force
Remove-Item -Recurse -Force $tmp
Write-Host "onnxruntime_providers_cuda.dll -> $((Resolve-Path $dest).Path)"
