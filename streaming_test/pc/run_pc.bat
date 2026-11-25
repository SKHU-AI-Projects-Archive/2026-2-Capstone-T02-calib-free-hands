@echo off
rem Double-click to start the PC receiver (MediaMTX) - see run_pc.py / run_config.yaml
cd /d "%~dp0"

rem Pick one Python and use it for the check, the install and the launcher.
where python >nul 2>nul
if %errorlevel%==0 (
    set "PY=python"
) else (
    set "PY=py -3"
)

%PY% -c "import yaml, segno" >nul 2>nul
if errorlevel 1 (
    echo pyyaml/segno not found, installing them with: %PY% -m pip install pyyaml segno
    %PY% -m pip install pyyaml segno
    if errorlevel 1 (
        echo.
        echo Could not install pyyaml/segno. Install them manually: %PY% -m pip install pyyaml segno
        goto :done
    )
)

%PY% run_pc.py %*

:done
echo.
pause
