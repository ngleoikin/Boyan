@echo off
setlocal
chcp 65001 >nul
if "%XIAOZHI_BASE%"=="" set XIAOZHI_BASE=http://127.0.0.1:8768
echo BASE=%XIAOZHI_BASE%
python test_mcp_tools.py
set rc=%ERRORLEVEL%
if %rc%==0 (
  echo ✅ 所有用例通过
) else (
  echo ❌ 存在失败用例，详见 test_report.txt / test_report.json
)
exit /b %rc%
