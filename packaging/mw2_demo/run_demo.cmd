@echo off
rem Muscle Wrapping 2.x demo: OsimMuscleGeneratorTool (OpenSim 4.5) on the output of msk-pipe.
rem   run_demo.cmd              the bundled example (example\)
rem   run_demo.cmd RUN          a msk-pipe run folder or its 06_mw2_input folder
rem   run_demo.cmd RUN WORK     work folder (default: output\<run name> next to this file)
rem The input is copied to the work folder; Muscle Wrapping writes its results there.
setlocal EnableExtensions DisableDelayedExpansion
set "DEMO=%~dp0"
set "DEMO=%DEMO:~0,-1%"
set "RC=0"

rem pause at the end only when started by a double-click (cmd /c "...\run_demo.cmd")
set "PAUSE_AT_END="
setlocal EnableDelayedExpansion
set "CL=!CMDCMDLINE!"
if not "!CL:%~nx0=!"=="!CL!" (endlocal & set "PAUSE_AT_END=1") else endlocal
if defined MW2_DEMO_NO_PAUSE set "PAUSE_AT_END="

set "IN=%~1"
if "%IN%"=="" set "IN=%DEMO%\example"
for %%I in ("%IN%") do set "IN=%%~fI"
if exist "%IN%\06_mw2_input\setup_MuscleGeneratorTool.xml" set "IN=%IN%\06_mw2_input"
if not exist "%IN%\setup_MuscleGeneratorTool.xml" (
    echo error: no setup_MuscleGeneratorTool.xml in "%IN%"
    echo        give a msk-pipe run folder or its 06_mw2_input folder
    set "RC=2"
    goto end
)

rem run name: the run folder (parent of 06_mw2_input), else the folder itself
for %%I in ("%IN%") do set "NAME=%%~nxI"
if /i "%NAME%"=="06_mw2_input" for %%I in ("%IN%\..") do set "NAME=%%~nxI"
set "WORK=%~2"
if "%WORK%"=="" set "WORK=%DEMO%\output\%NAME%"
for %%I in ("%WORK%") do set "WORK=%%~fI"
if /i "%WORK%"=="%IN%" (
    echo error: the work folder must not be the input folder
    set "RC=2"
    goto end
)

echo Input:  %IN%
echo Output: %WORK%
"%SystemRoot%\System32\robocopy.exe" "%IN%" "%WORK%" /E /XD analysis fibres /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 (
    echo error: copying the input failed
    set "RC=1"
    goto end
)

rem results of an earlier run in the work folder are removed first
set "OSIM="
for /f "tokens=1-3 delims=<>" %%A in ('findstr /c:"<output_model_file>" "%WORK%\setup_MuscleGeneratorTool.xml"') do (
    if "%%A"=="output_model_file" (set "OSIM=%%B") else set "OSIM=%%C"
)
if defined OSIM set "OSIM=%OSIM:/=\%"
if defined OSIM if exist "%WORK%\%OSIM%" del /q "%WORK%\%OSIM%"
if exist "%WORK%\analysis" rd /s /q "%WORK%\analysis"
if exist "%WORK%\fibres" rd /s /q "%WORK%\fibres"
mkdir "%WORK%\analysis"

echo Running Muscle Wrapping (a few minutes; log: mw2.log) ...
set "PATH=%DEMO%\bin;%PATH%"
pushd "%WORK%"
"%DEMO%\bin\opensim-cmd.exe" -L "%DEMO%\bin\OsimMuscleGeneratorTool.dll" run-tool setup_MuscleGeneratorTool.xml > mw2.log 2>&1
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo error: Muscle Wrapping failed with exit code %RC%, see "%WORK%\mw2.log"
    goto end
)
if not defined OSIM goto done
if not exist "%WORK%\%OSIM%" (
    echo error: Muscle Wrapping wrote no %OSIM%, see "%WORK%\mw2.log"
    set "RC=1"
    goto end
)
set "FIBRES=0"
for /f %%N in ('findstr /c:"Muscle name=" "%WORK%\%OSIM%" ^| "%SystemRoot%\System32\find.exe" /c /v ""') do set "FIBRES=%%N"
echo Model with %FIBRES% fibres: %WORK%\%OSIM%

:done
echo Fibre lengths and moment arms: %WORK%\analysis

:end
if defined PAUSE_AT_END pause
endlocal & exit /b %RC%
