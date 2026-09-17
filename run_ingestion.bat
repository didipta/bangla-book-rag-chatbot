@echo off
setlocal
python -m app.crawler
if errorlevel 1 exit /b 1
python -m app.ingest
if errorlevel 1 exit /b 1
echo.
echo Ingestion completed successfully.
pause
