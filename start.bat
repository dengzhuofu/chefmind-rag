@echo off
chcp 65001 >nul
title ChefMind 启动器

echo ========================================
echo   ChefMind - 私厨大脑 启动中...
echo ========================================
echo.

:: 启动后端
echo [1/2] 启动后端 (FastAPI)...
start "ChefMind Backend" cmd /k "cd /d %~dp0backend && python -m app.main"

:: 等待2秒让后端先启动
timeout /t 2 /nobreak >nul

:: 启动前端
echo [2/2] 启动前端 (Next.js)...
start "ChefMind Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   启动完成！
echo   后端: http://localhost:8000
echo   前端: http://localhost:3000
echo ========================================
echo.
pause
