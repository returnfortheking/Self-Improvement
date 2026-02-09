@echo off
REM ====================================
REM OpenCode Server启动脚本（Windows改进版）
REM ====================================
echo.
echo [INFO] 启动OpenCode HTTP Server
echo [INFO] 项目目录：%~dp0
echo [INFO] 工作目录：%cd%
echo.

REM 切换到项目目录
cd /d "%~dp0"
if errorlevel neq 0 (
    echo [ERROR] 无法切换到项目目录
    pause
    exit /b 1
)

echo [INFO] 切换到工作目录成功
cd "%~dp0"

REM 检查opencode是否安装
where opencode >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] opencode未安装，请先安装：
    echo   curl -fsSL https://opencode.ai/install ^| bash
    echo.
    pause
    exit /b 1
)

echo [INFO] opencode已安装，版本检查完成
echo.

REM 配置信息
set SERVER_PORT=4096
set SERVER_HOST=0.0.0.0
set CORS_ORIGINS=http://localhost:5173 http://localhost:5174

REM 检查端口是否已被占用
netstat -ano | findstr ":%SERVER_PORT% " | findstr "LISTENING"
if %errorlevel% equ 0 (
    echo [WARNING] 端口%SERVER_PORT%已被占用，正在尝试释放...
    for /f "tokens=5" %%a in ('netstat -ano -n ^|^.*:%SERVER_PORT% $') do (
        if %%a equ LISTENING (
            echo [INFO] 正在终止占用端口的进程...
            for /f "tokens=2" %%b in (%%a) do taskkill /F /PID %%b
        )
    )
    )
    timeout /t 3 >nul 2>&1
    netstat -ano | findstr ":%SERVER_PORT%" | findstr "LISTENING"
    if %errorlevel% neq 0 (
        echo [ERROR] 无法释放端口，请手动关闭占用端口的程序
        pause
        exit /b 1
    )
    echo [INFO] 端口检查完成
)

echo.
echo [INFO] 服务器配置：
echo   端口：%SERVER_PORT%
echo   主机：%SERVER_HOST%
echo   CORS：%CORS_ORIGINS%
echo.
echo ====================================
echo [INFO] 启动OpenCode Server...
echo ====================================
echo.

REM 启动OpenCode Server（后台运行）
start "" opencode serve --port %SERVER_PORT% --hostname %SERVER_HOST% --cors %CORS_ORIGINS%
if %errorlevel% neq 0 (
    echo [ERROR] 启动失败
    pause
    exit /b 1
)

echo.
echo ====================================
echo [INFO] OpenCode Server已在后台运行
echo 服务器地址：http://%SERVER_HOST%:%SERVER_PORT%/doc
echo ====================================
echo.
echo [提示] 服务器将在当前窗口运行
echo [提示] 关闭此窗口将停止服务器
echo [提示] 可以在另一个终端窗口查看API文档：http://%SERVER_HOST%:%SERVER_PORT%/doc
echo ====================================
echo.

REM 等待用户按键
echo [INFO] 按任意键停止服务器...
pause

REM 清理
set SERVER_PORT=
set SERVER_HOST=
set CORS_ORIGINS=
