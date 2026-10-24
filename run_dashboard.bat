@echo off
setlocal
pushd "%~dp0"

set "PYTHON=.venv-py313\Scripts\python.exe"
if not exist "%PYTHON%" (
    py -3.13 -m venv .venv-py313
    if errorlevel 1 goto setup_failed
)

"%PYTHON%" -c "import torch, fastapi, uvicorn, transformers" >nul 2>nul
if errorlevel 1 (
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
)

"%PYTHON%" start_app.py
if errorlevel 1 goto run_failed
popd
exit /b 0

:setup_failed
echo Dashboard setup failed. Check the Python 3.13 installation and internet connection.
pause
popd
exit /b 1

:run_failed
echo Dashboard exited with an error. Review the terminal output above.
pause
popd
exit /b 1
