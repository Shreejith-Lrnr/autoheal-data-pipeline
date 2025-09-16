import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING", 
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ProductionPipelineDB;Trusted_Connection=yes;")
    
    # AI Configuration  
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Make model configurable
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # Alternative AI Providers (for future use)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Optional
    
    # Available Groq Models
    AVAILABLE_GROQ_MODELS = [
        "llama3-8b-8192",      # Default - Fast and efficient
        "llama3-70b-8192",     # Larger, more capable
        "mixtral-8x7b-32768",  # Different architecture
        "gemma-7b-it",         # Smaller, faster
        "llama2-70b-4096"      # Alternative Llama model
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
