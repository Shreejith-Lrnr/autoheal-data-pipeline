## 1. **Clone the Repository**  
```bash  
git clone <repository-url>  
cd production_agentic_pipeline  
```

## 2. **Create Python Virtual Environment**  
```bash  
# Windows  
python -m venv venv  
venv\Scripts\activate

# macOS/Linux  
python3 -m venv venv  
source venv/bin/activate  
```

## 3. **Install Dependencies**  
```bash  
pip install -r requirements.txt  
```

## 4. **Set Up Environment Variables**

Create a `.env` file in the project root (copy from `.env.example`):

```env  
# Database Connection  
DATABASE_CONNECTION_STRING=DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ProductionPipelineDB;Trusted_Connection=yes;

# Groq AI API  
GROQ_API_KEY=your_groq_api_key_here

# Optional: OpenAI (if needed)  
OPENAI_API_KEY=your_openai_key_here

# Optional: Anthropic (if needed)  
ANTHROPIC_API_KEY=your_anthropic_key_here  
```

### **To Get GROQ_API_KEY:**  
1. Go to [console.groq.com](https://console.groq.com)  
2. Sign up/Log in  
3. Create an API key  
4. Copy and paste it in `.env`

## 5. **Set Up SQL Server Database**

### **Option A: Using SQL Server Management Studio (SSMS)**  
```sql  
-- Open SSMS and run:  
CREATE DATABASE ProductionPipelineDB;  
```

### **Option B: Using Command Line**  
```bash  
sqlcmd -S localhost -U sa -P YourPassword -Q "CREATE DATABASE ProductionPipelineDB;"  
```

**Note:** Tables will be created automatically when the app first runs (via database.py's `ensure_tables_exist()` method)

## 6. **Verify Database Connection String**

Edit config.py if your database setup is different:

```python  
# config.py - Line 8-9  
DATABASE_CONNECTION_STRING = os.getenv(  
    "DATABASE_CONNECTION_STRING",   
    "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ProductionPipelineDB;Trusted_Connection=yes;"  
)  
```

**Common variations:**  
```python  
# If using SQL Server authentication instead of Windows:  
"DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ProductionPipelineDB;UID=sa;PWD=your_password;"

# If SQL Server is on a remote machine:  
"DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.1.100;DATABASE=ProductionPipelineDB;Trusted_Connection=yes;"

# If using different port:  
"DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost,1433;DATABASE=ProductionPipelineDB;Trusted_Connection=yes;"  
```

## 7. **Install ODBC Driver (if needed)**

### **Windows:**  
```bash  
# Already installed on most Windows systems with SQL Server  
# If missing, download from: [https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server "https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")  
```

### **macOS:**  
```bash  
brew install unixodbc  
brew install microsoft-odbc-driver-for-sql-server  
```

### **Linux (Ubuntu/Debian):**  
```bash  
curl [https://packages.microsoft.com/keys/microsoft.asc](https://packages.microsoft.com/keys/microsoft.asc "https://packages.microsoft.com/keys/microsoft.asc") | apt-key add -  
curl [https://packages.microsoft.com/config/ubuntu/20.04/prod.list](https://packages.microsoft.com/config/ubuntu/20.04/prod.list "https://packages.microsoft.com/config/ubuntu/20.04/prod.list") > /etc/apt/sources.list.d/mssql-release.list  
sudo apt-get update  
sudo apt-get install msodbcsql17  
```

## 8. **Run the Application**

```bash  
streamlit run streamlit_app.py  
```

The app will open at: **http://localhost:8501**

---

## **Code Changes Needed (If Any)**

### **If NOT using SQL Server (Using SQLite instead):**

Edit config.py:  
```python  
# Change DATABASE_CONNECTION_STRING to use SQLite  
import sqlite3  
DATABASE_CONNECTION_STRING = "sqlite:///pipeline_data.db"  
```

Edit database.py - Replace pyodbc with sqlite3:  
```python  
import sqlite3  
from contextlib import contextmanager

class ProductionDatabase:  
    def __init__(self):  
        self.db_file = "pipeline_data.db"  
        self.ensure_tables_exist()  
  
    def get_connection(self):  
        return sqlite3.connect(self.db_file)  
```

### **If Running on Linux/macOS:**

No major changes needed, but verify:  
1. **Python version:** Python 3.8+ required  
2. **Streamlit cache directory:** May need to set manually  
```bash  
export STREAMLIT_SERVER_HEADLESS=true  
streamlit run streamlit_app.py  
```

---

## **Troubleshooting Checklist**

| Issue | Solution |  
|-------|----------|  
| `ModuleNotFoundError: No module named 'pyodbc'` | Run `pip install -r requirements.txt` |  
| `GROQ API Key Error` | Check `.env` file, verify API key is valid at console.groq.com |  
| `Database connection failed` | Check `DATABASE_CONNECTION_STRING` in `.env`, ensure SQL Server is running |  
| `ODBC Driver not found` | Install ODBC Driver 17 for your OS |  
| `Port 8501 already in use` | Run `streamlit run streamlit_app.py --server.port 8502` |  
| `SSL/Certificate errors` | Update certificates: `pip install --upgrade certifi` |

---

## **File Structure After Setup**

```  
production_agentic_pipeline/  
├── .env                          # ← Create this (secret, don't commit)  
├── .env.example                  # ← Template (safe to commit)  
├── streamlit_app.py  
├── agents.py  
├── database.py  
├── config.py  
├── pipeline_engine.py  
├── real_time_monitor.py  
├── agent_learning.py  
├── requirements.txt  
├── README.md  
└── sample_data/  
    ├── sales_data.csv  
    ├── customer_data.csv  
    └── inventory_data.csv  
```

---

## **Quick Start Script**

Create `setup.sh` (macOS/Linux) or `setup.bat` (Windows) for one-command setup:

**setup.sh:**  
```bash  
#!/bin/bash  
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  
echo "Setup complete! Run: streamlit run streamlit_app.py"  
```

**setup.bat (Windows):**  
```batch  
@echo off  
python -m venv venv  
call venv\Scripts\activate.bat  
pip install -r requirements.txt  
echo Setup complete! Run: streamlit run streamlit_app.py  
```

Run with: `bash setup.sh` or `setup.bat`

Download ODBC Driver for SQL Server - ODBC Driver for SQL Server

Download the Microsoft ODBC Driver for SQL Server to develop native-code applications that connect to SQL Server and Azure SQL Database.
