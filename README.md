# Production-Ready Agentic AI Data Pipeline

  

This system demonstrates a real, production-ready agentic AI pipeline for enterprise data processing, featuring:

  

- Real MS SQL Server integration

- File upload for actual data

- True AI agent collaboration

- Human-in-the-loop approvals

- Complete audit trail

- Scalable architecture

  

## Project Structure

  

```

production_agentic_pipeline/

├── config.py

├── database.py

├── agents.py

├── pipeline_engine.py

├── streamlit_app.py

├── requirements.txt

├── .env

├── sample_data/

│   ├── sales_data.csv

│   ├── customer_data.csv

│   └── inventory_data.csv

└── README.md

```

  

## Setup Instructions

0. **Pull Code from Github (within terminal):**

```
git clone https://github.com/Shreejith-Lrnr/autoheal-data-pipeline.git

```

```
cd autoheal-data-pipeline

```

2. **Install Requirements:**

```
   
   pip install -r requirements.txt


```

2. **Setup Database:**

   - Open SQL Server Management Studio

   - Create database `ProductionPipelineDB`

   - Tables will be created automatically

3. **Configure Environment:**

   - Update `.env` with your database connection and Groq API key

4. **Run Application:**

```

   streamlit run streamlit_app.py

```

5. **Access Application:**

   - Open browser to `http://localhost:8501`

   - Upload your CSV/Excel files

   - Watch AI agents process your data

  

See the code for more details and customization options.

  

## How to run this project after pulling from GitHub (step-by-step, plain language)

  

Follow these exact steps after you clone or pull this repository so the whole app (the Streamlit module and the new frontend "Module Nexus") runs correctly.

  

1) Open a terminal (Command Prompt) and change to the project folder

  

```

cd C:\Users\<your-username>\Projects\Documents\UST\trial\production_agentic_pipeline

```

  

Replace `<your-username>` with your Windows username, or use the path where you cloned the repo. You must be inside the `production_agentic_pipeline` folder for the commands below.

  

2) (Optional but recommended) Create and activate a Python virtual environment

  

```

python -m venv .venv

.venv\Scripts\activate

```

  

If your system uses `python3` instead of `python`, run `python3 -m venv .venv`. After activation your prompt should show `(.venv)`.

  

3) Install Python dependencies for the Streamlit app

  

```

pip install -r requirements.txt

```

  

This installs all Python packages the Streamlit app needs.

  

4) Prepare environment variables

  

- Open the `.env` file in the `production_agentic_pipeline` folder and set the values for any required keys (database connection, API keys). If you don't have a database or API keys for now, you can leave them blank but some features may not work.

  

5) Start the Streamlit app (the core module)

  

Keep this terminal open and run:

  

```

streamlit run streamlit_app.py

```

  

This starts the Streamlit app on `http://localhost:8501` by default. Leave this running while you use the hub.

  

6) Start the Module Nexus frontend (the hub) in a new terminal window

  

Open a second Command Prompt window and change to the `module-nexus` folder inside the project:

  

```

cd C:\Users\<your-username>\Projects\Documents\production_agentic_pipeline\module-nexus

```

  

7) Install Node dependencies (first time only)

  

If you haven't already installed dependencies in this folder, run:

  

```

npm install

```

  
  

8) Start the frontend dev server

  

If you used `start-dev.bat` the dev server starts automatically. Otherwise run:

  

```

npm run dev

```

  

Vite will print a local URL (commonly `http://localhost:5173`) — open that in your browser.

  

9) Use the hub to open the Streamlit module

  

In the Module Nexus UI, click the `Autoheal Pipeline` module card. It opens the Streamlit app in a new browser tab at `http://localhost:8501`.

  

Troubleshooting tips (common issues and fixes)

  

- "Command not found" for `python` or `npm`: make sure Python and Node are installed and on your PATH. Use `python --version` and `node --version` to check.

- If Vite complains about ESM/CJS issues: ensure Node 18+ is installed. Remove and reinstall node modules if needed:

  

```

rmdir /s /q node_modules

npm install

```

  

- If Git shows already-tracked noisy files (like `node_modules`), run these commands to stop tracking and remove them from the index (do this from the `production_agentic_pipeline` root):

  

```

git rm -r --cached module-nexus/node_modules

git commit -m "Remove node_modules from repo"

```

  

- If the Streamlit app fails to start due to missing database/config: open `.env` and add correct connection info or run in a mode that uses sample data.

  

If you'd like, I can add a single script that starts both servers for development, or create a quick checklist you can follow. Let me know which you prefer.
