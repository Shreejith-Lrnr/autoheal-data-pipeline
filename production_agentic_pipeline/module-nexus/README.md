# Module Nexus — Agentic Modules Hub

This is a small React frontend that acts as a central site for accessing your agentic modules (referred to as "modules" inside this app). It currently surfaces a single module — the `Autoheal Pipeline` (your Streamlit app) — and opens it in a new tab.

Quick start (Windows / cmd.exe):

1. Install frontend dependencies

```
cd production_agentic_pipeline\module-nexus
npm install
```

2. Run the frontend (Vite dev server)

Option A — one-line helper (recommended):

```
start-dev.bat
```

Option B — manual:

```
npm run dev
```

3. Start the Streamlit module from the repo root (if not already running)

```
cd ..\
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

4. In the hub UI, click the `Autoheal Pipeline` card to open the module in a new tab. The hub expects the module to be served at `http://localhost:8501`.

Notes & troubleshooting:

- Node: Use a modern Node.js (Node 18 or newer recommended). Check with `node --version`.
- If you installed `node_modules` before we set the frontend to ESM mode, remove and reinstall dependencies:

```
rmdir /s /q node_modules
npm install
```

- The hub supports a light/dark theme toggle and uses `localStorage` to remember the theme.
- Add more modules by editing `src/App.jsx` and adding entries to the `modules` array.

If you want, I can create a production build and add hosting instructions.
