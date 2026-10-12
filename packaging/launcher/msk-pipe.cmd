@echo off
rem msk-pipe - double-click to start the graphical user interface.
setlocal
call "%~dp0bin\_setup.cmd"
if errorlevel 1 (
  pause
  exit /b 1
)
rem run folders go to Documents\msk-pipe\runs (the GUI remembers another choice)
set "WORK=%USERPROFILE%\Documents\msk-pipe"
if not exist "%USERPROFILE%\Documents\" set "WORK=%USERPROFILE%\msk-pipe"
if not exist "%WORK%" mkdir "%WORK%"
cd /d "%WORK%"
start "msk-pipe" "%MSKPIPE_PYTHONW%" -m mskpipe.gui
exit /b 0
