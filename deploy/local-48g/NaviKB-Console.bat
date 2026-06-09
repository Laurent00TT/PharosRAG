@echo off
REM NaviKB local 4090-48G control console — double-click to run.
chcp 65001 >nul
title NaviKB Local 4090-48G Console
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0navikb-console.ps1"
echo.
echo (console closed)
pause
