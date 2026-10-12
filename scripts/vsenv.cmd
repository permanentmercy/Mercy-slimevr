@echo off
REM Wrapper: sets up MSVC x64 environment for a VS install at a non-standard path,
REM then runs the command passed as arguments.
call "D:\software\VS\product\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
  echo [vsenv] vcvarsall failed
  exit /b 1
)
%*
exit /b %ERRORLEVEL%
