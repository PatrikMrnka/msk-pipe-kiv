@echo off
rem msk-pipe release: prepare the environment (called by msk-pipe.cmd and mskpipe.cmd).
rem First start (or after the folder was moved): unpack environment.tar into env\.
rem Sets MSKPIPE_ROOT, MSKPIPE_PYTHON, MSKPIPE_PYTHONW and the variables of the release.
chcp 65001 >nul
for %%I in ("%~dp0..") do set "MSKPIPE_ROOT=%%~fI"
set "MSKPIPE_ENV=%MSKPIPE_ROOT%\env"
set "MSKPIPE_PYTHON=%MSKPIPE_ENV%\python.exe"
set "MSKPIPE_PYTHONW=%MSKPIPE_ENV%\pythonw.exe"
set "MSKPIPE_VERSION="
set /p MSKPIPE_VERSION=<"%MSKPIPE_ROOT%\version.txt"
set "DONE="
if exist "%MSKPIPE_ENV%\.mskpipe-unpacked" set /p DONE=<"%MSKPIPE_ENV%\.mskpipe-unpacked"
rem the environment is bound to its folder (conda prefix): unpack again after a move
if not "%DONE%"=="%MSKPIPE_VERSION%*%MSKPIPE_ROOT%" call :unpack || exit /b 1

call "%MSKPIPE_ROOT%\activate.bat" >nul
rem the release is self-contained: no user site-packages or foreign PYTHONPATH
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONUTF8=1"
if not defined MSKPIPE_ATLAS_DIR if exist "%MSKPIPE_ROOT%\atlas\lhdl\atlas.json" set "MSKPIPE_ATLAS_DIR=%MSKPIPE_ROOT%\atlas\lhdl"
exit /b 0

:unpack
echo msk-pipe %MSKPIPE_VERSION%: preparing the environment (first start, 1-3 minutes) ...
call :check_path || exit /b 1
if exist "%MSKPIPE_ENV%" rmdir /s /q "%MSKPIPE_ENV%"
if exist "%MSKPIPE_ENV%" (
  echo ERROR: cannot remove "%MSKPIPE_ENV%" - close msk-pipe and try again.
  exit /b 1
)
"%MSKPIPE_ROOT%\bin\pixi-unpack.exe" "%MSKPIPE_ROOT%\environment.tar" -o "%MSKPIPE_ROOT%" -e env -s cmd -q
if errorlevel 1 (
  echo ERROR: unpacking environment.tar failed. Is there enough free disk space ^(~5 GB^)?
  exit /b 1
)
if not exist "%MSKPIPE_PYTHON%" (
  echo ERROR: the environment has no python.exe after unpacking.
  exit /b 1
)
>"%MSKPIPE_ENV%\.mskpipe-unpacked" echo %MSKPIPE_VERSION%*%MSKPIPE_ROOT%
echo Environment ready: %MSKPIPE_ENV%
exit /b 0

:check_path
rem Windows limits paths to 260 characters unless long paths are enabled.
set "MAXREL=0"
set /p MAXREL=<"%MSKPIPE_ROOT%\bin\maxpath.txt"
call :strlen MSKPIPE_ROOT ROOTLEN
set /a "TOTAL=ROOTLEN+MAXREL+1"
if %TOTAL% LSS 260 exit /b 0
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | find "0x1" >nul
if not errorlevel 1 exit /b 0
echo ERROR: the folder path is too long for Windows ^(%TOTAL% of 259 characters^).
echo Move the msk-pipe folder to a short path, e.g. C:\msk-pipe, and start it again.
exit /b 1

:strlen
rem %1 = name of the variable to measure, %2 = name of the result variable
setlocal EnableDelayedExpansion
set "s=!%~1!#"
set "len=0"
for %%P in (4096 2048 1024 512 256 128 64 32 16 8 4 2 1) do (
  if "!s:~%%P,1!" NEQ "" (
    set /a "len+=%%P"
    set "s=!s:~%%P!"
  )
)
endlocal & set "%~2=%len%"
exit /b 0
