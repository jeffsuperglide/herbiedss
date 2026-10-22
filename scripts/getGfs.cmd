@echo off
setlocal EnableDelayedExpansion

if "%~1"=="" goto usage

set "start=%~1"
set "end=%~2"
if "%~2"=="" set "end=%start%"

@REM Integers only, no leading zeros
echo %start%| findstr /r "^[1-9][0-9]*$" >nul || goto usage
echo %end%| findstr /r "^[1-9][0-9]*$" >nul || goto usage

if %start% LSS 1 goto usage
if %end% GTR 384 goto usage
if %start% GTR %end% goto usage

@REM Hours above 120 must be on the 3-hour step (123, 126, ... 384)
if %start% GTR 120 (
    set /a "mod=%start% %% 3"
    if !mod! NEQ 0 goto usage_step
)
if %end% GTR 120 (
    set /a "mod=%end% %% 3"
    if !mod! NEQ 0 goto usage_step
)

set "numbers="
set "separator="

set "HERBIEDSS=C:\Users\Public\RTS\.venv\Scripts\herbiedss"
set "PY=C:\Users\Public\RTS\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: Python not found in .venv
  pause
  exit /b 1
)

rem Portion within hours 1-120 as a range
if %start% LEQ 120 (
    set "rangeEnd=%end%"
    if !rangeEnd! GTR 120 set "rangeEnd=120"

    if %start% EQU !rangeEnd! (
        set "numbers=%start%"
    ) else (
        set "numbers=%start%-!rangeEnd!"
    )
    set "separator=,"
)

rem Append 3-hourly hours from 123 through 384
for /L %%N in (123,3,384) do (
    if %%N GEQ %start% if %%N LEQ %end% (
        set "numbers=!numbers!!separator!%%N"
        set "separator=,"
    )
)

for /f "usebackq delims=" %%I in (`%PY% -c "from datetime import datetime, timezone;from zoneinfo import ZoneInfo;d=datetime.now(ZoneInfo('America/Panama'));d=d.replace(hour=(d.hour//6)*6,minute=0,second=0,microsecond=0);print(d.replace(tzinfo=None).isoformat(timespec='minutes'))"`) do set "UTC_CYCLE=%%I"

if not exist "%HERBIEDSS%.exe" (
  echo ERROR: Script not found: %HERBIEDSS%
  pause
  exit /b 1
)

echo ==== Executing herbiedss for model cycle %UTC_CYCLE% ====
echo ==== Forecast hours !numbers! ====
%HERBIEDSS% dss --date %UTC_CYCLE% ^
--model gfs --product pgrb2.0p25 ^
--fxx !numbers! --subset ":PRATE:surface:(?:[0-9]+ hour fcst|anl):" ^
--boundary "-81.6251 6.8749 -77.8749 10.6251" --bbox-epsg 4326 ^
--dssfile "C:\Users\Public\RTS\database\precipgrids.dss" ^
--bpart "PANAMA" ^
--fpart "GFS-PGRB2.0P25"

exit /b 0

:usage_step
>&2 echo Hours above 120 must be 123 or higher and a multiple of 3 (123, 126, ... 384).
exit /b 1

:usage
>&2 echo Usage: %~nx0 startHour [endHour]
>&2 echo Integers without leading zeros: 1 LEQ startHour LEQ endHour LEQ 384.
>&2 echo If endHour is omitted, only startHour is run.
exit /b 1
