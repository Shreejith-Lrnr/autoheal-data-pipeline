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
│   ├── sales_data.csv
│   ├── customer_data.csv
│   └── inventory_data.csv
└── README.md
```

## Setup Instructions

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Database:**
   - Open SQL Server Management Studio
   - Create database `ProductionPipelineDB`
   - Tables will be created automatically
3. **Configure Environment:**
   - Update `.env` with your database connection and Groq API key
4. **Run Application:**
   ```bash
   streamlit run streamlit_app.py
   ```
5. **Access Application:**
   - Open browser to `http://localhost:8501`
   - Upload your CSV/Excel files
   - Watch AI agents process your data

See the code for more details and customization options.
