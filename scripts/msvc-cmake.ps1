param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Binary,
    [string]$BuildType = "Release",
    [switch]$ConfigureOnly
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$msvc  = "D:\software\VS\product\VC\Tools\MSVC\14.51.36231"
$sdk   = "C:\Program Files (x86)\Windows Kits\10"
$sdkv  = "10.0.26100.0"
$cmake = "D:\software\VS\product\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "D:\software\VS\product\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

$env:PATH    = "$msvc\bin\Hostx64\x64;$sdk\bin\$sdkv\x64;$env:PATH"
$env:INCLUDE = "$msvc\include;$sdk\Include\$sdkv\ucrt;$sdk\Include\$sdkv\um;$sdk\Include\$sdkv\shared;$sdk\Include\$sdkv\winrt;$sdk\Include\$sdkv\cppwinrt"
$env:LIB     = "$msvc\lib\x64;$sdk\Lib\$sdkv\ucrt\x64;$sdk\Lib\$sdkv\um\x64"

Write-Host "cl    : $((Get-Command cl.exe).Source)"
Write-Host "cmake : $cmake"

$cmakeArgs = @(
    "-S", $Source,
    "-B", $Binary,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$msvc\bin\Hostx64\x64\cl.exe",
    "-DCMAKE_CXX_COMPILER=$msvc\bin\Hostx64\x64\cl.exe",
    "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$Binary\$BuildType"
)

Write-Host "--- configure ---"
& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

if (-not $ConfigureOnly) {
    Write-Host "--- build ---"
    & $cmake --build $Binary --config $BuildType
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }
}
Write-Host "OK"
