@echo off
REM Helper to start Module Nexus frontend on Windows (cmd.exe)
pushd %~dp0

if not exist node_modules (
  echo node_modules not found — installing dependencies...
  npm install || (
    echo npm install failed. Ensure Node.js (>=18) and npm are installed.
    popd
    exit /b 1
  )
)

echo Starting Vite dev server...
npm run dev

popd
