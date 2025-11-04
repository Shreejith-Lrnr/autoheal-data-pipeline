import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING", 
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ProductionPipelineDB;Trusted_Connection=yes;")
    
    # AI Configuration  
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Updated default model
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # Alternative AI Providers (for future use)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Optional
    
    # Available Groq Models (Updated for current API)
    AVAILABLE_GROQ_MODELS = [
        "llama-3.1-8b-instant",    # Fast and efficient
        "llama-3.1-70b-versatile", # Larger, more capable
        "llama-3.1-405b-instruct", # Largest model
        "llama3-70b-8192",         # Fallback
        "gemma2-9b-it"             # Alternative
    ]
    
    # Pipeline Configuration
    BATCH_SIZE = 1000
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_FILE_TYPES = ['csv', 'xlsx', 'json']
    
    # Data Quality Rules
    QUALITY_RULES = {
        'null_threshold': 0.05,  # 5% null values max
        'duplicate_threshold': 0.02,  # 2% duplicates max
        'outlier_std_threshold': 3.0  # 3 standard deviations
    }
    
    @classmethod
    def validate_config(cls):
        if not cls.GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not found. System will use fallback responses.")
            return False
        return True
