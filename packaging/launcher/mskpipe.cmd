@echo off
rem msk-pipe command line, e.g.:  mskpipe.cmd run ct.nii.gz -m ct
rem                               mskpipe.cmd setup status
setlocal
call "%~dp0bin\_setup.cmd"
if errorlevel 1 exit /b 1
"%MSKPIPE_PYTHON%" -m mskpipe %*
exit /b %ERRORLEVEL%
