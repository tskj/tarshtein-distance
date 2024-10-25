@echo off
REM Run zig build install
zig build install

REM Check if the build was successful
IF NOT EXIST "zig-out\lib\levvy.dll" (
    echo Build failed or levvy.dll not found.
    exit /b 1
)

REM Copy levvy.dll to the target directory
set "SOURCE=zig-out\lib\levvy.dll"
set "DEST=C:\Users\tarje\AppData\Local\nvim\zig\levvy.dll"

IF NOT EXIST "%DEST%" mkdir "%DEST%\.."
copy "%SOURCE%" "%DEST%"

IF %ERRORLEVEL% EQU 0 (
    echo levvy.dll successfully copied to %DEST%
) ELSE (
    echo Failed to copy levvy.dll
)

