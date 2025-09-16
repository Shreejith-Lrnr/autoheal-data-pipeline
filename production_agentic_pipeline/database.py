import pyodbc
import pandas as pd
import json
from datetime import datetime
from config import Config

class ProductionDatabase:
    def __init__(self):
        self.connection_string = Config.DATABASE_CONNECTION_STRING
        self.ensure_tables_exist()
    
    def get_connection(self):
        """Get database connection with error handling"""
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")
    
    def ensure_tables_exist(self):
        """Create necessary tables if they don't exist"""
        create_tables_sql = """
        -- Main data processing table
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ProcessedData' AND xtype='U')
        CREATE TABLE ProcessedData (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            DatasetName NVARCHAR(255),
            OriginalData NTEXT,
            ProcessedData NTEXT,
            DataQualityScore DECIMAL(5,2),
            ProcessingTimestamp DATETIME DEFAULT GETDATE(),
            Status NVARCHAR(50)
        );
        
        -- Data quality issues table
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='DataQualityIssues' AND xtype='U')
        CREATE TABLE DataQualityIssues (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            DatasetID INT FOREIGN KEY REFERENCES ProcessedData(ID),
            IssueType NVARCHAR(100),
            IssueDescription NTEXT,
            AffectedRows INT,
            Severity NVARCHAR(20),
            DetectedTimestamp DATETIME DEFAULT GETDATE()
        );
        
        -- Agent actions log
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AgentActions' AND xtype='U')
        CREATE TABLE AgentActions (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            DatasetID INT FOREIGN KEY REFERENCES ProcessedData(ID),
            AgentName NVARCHAR(100),
            ActionType NVARCHAR(100),
            ActionDetails NTEXT,
            ExecutionStatus NVARCHAR(50),
            HumanApproval NVARCHAR(20),
            ActionTimestamp DATETIME DEFAULT GETDATE()
        );
        
        -- Pipeline monitoring table
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='PipelineMonitoring' AND xtype='U')
        CREATE TABLE PipelineMonitoring (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            PipelineName NVARCHAR(255),
            Status NVARCHAR(50),
            StartTime DATETIME,
            EndTime DATETIME,
            RecordsProcessed INT,
            IssuesFound INT,
            IssuesResolved INT,
            MonitoringData NTEXT
        );
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(create_tables_sql)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Table creation warning: {e}")
    
    def store_dataset(self, dataset_name, original_data, processed_data=None, quality_score=0):
        """Store dataset in database"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ProcessedData (DatasetName, OriginalData, ProcessedData, DataQualityScore, Status)
                VALUES (?, ?, ?, ?, ?)
            """, dataset_name, original_data.to_json(), 
                processed_data.to_json() if processed_data is not None else None,
                quality_score, "PROCESSING")
            
            conn.commit()
            return cursor.execute("SELECT @@IDENTITY").fetchone()[0]
        finally:
            conn.close()
    
    def update_dataset_quality_score(self, dataset_id, quality_score, processed_data=None, status="COMPLETED"):
        """Update dataset with quality score and processed data"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if processed_data is not None:
                cursor.execute("""
                    UPDATE ProcessedData 
                    SET DataQualityScore = ?, ProcessedData = ?, Status = ?
                    WHERE ID = ?
                """, quality_score, processed_data.to_json(), status, dataset_id)
            else:
                cursor.execute("""
                    UPDATE ProcessedData 
                    SET DataQualityScore = ?, Status = ?
                    WHERE ID = ?
                """, quality_score, status, dataset_id)
            conn.commit()
        finally:
            conn.close()
    
    def get_processed_data(self, dataset_id):
        """Get processed data for a dataset"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT ProcessedData FROM ProcessedData WHERE ID = ?", dataset_id)
            result = cursor.fetchone()
            if result and result[0]:
                return pd.read_json(result[0])
            return None
        finally:
            conn.close()
    
    def log_quality_issue(self, dataset_id, issue_type, description, affected_rows, severity):
        """Log data quality issues"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO DataQualityIssues (DatasetID, IssueType, IssueDescription, AffectedRows, Severity)
                VALUES (?, ?, ?, ?, ?)
            """, dataset_id, issue_type, description, affected_rows, severity)
            conn.commit()
        finally:
            conn.close()
    
    def log_agent_action(self, dataset_id, agent_name, action_type, action_details, execution_status="PENDING"):
        """Log agent actions"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO AgentActions (DatasetID, AgentName, ActionType, ActionDetails, ExecutionStatus)
                VALUES (?, ?, ?, ?, ?)
            """, dataset_id, agent_name, action_type, action_details, execution_status)
            conn.commit()
            return cursor.execute("SELECT @@IDENTITY").fetchone()[0]
        finally:
            conn.close()
    
    def update_agent_action(self, action_id, execution_status, human_approval=None):
        """Update agent action status"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE AgentActions 
                SET ExecutionStatus = ?, HumanApproval = ?
                WHERE ID = ?
            """, execution_status, human_approval, action_id)
            conn.commit()
        finally:
            conn.close()
    
    def get_dataset_summary(self):
        """Get summary of all processed datasets"""
        conn = self.get_connection()
        try:
            return pd.read_sql("""
                SELECT 
                    pd.ID,
                    pd.DatasetName,
                    pd.DataQualityScore,
                    pd.Status,
                    pd.ProcessingTimestamp,
                    COUNT(dqi.ID) as IssuesFound
                FROM ProcessedData pd
                LEFT JOIN DataQualityIssues dqi ON pd.ID = dqi.DatasetID
                GROUP BY pd.ID, pd.DatasetName, pd.DataQualityScore, pd.Status, pd.ProcessingTimestamp
                ORDER BY pd.ProcessingTimestamp DESC
            """, conn)
        finally:
            conn.close()
    
    def get_agent_actions(self, dataset_id=None):
        """Get agent actions log"""
        conn = self.get_connection()
        try:
            if dataset_id is None:
                query = """
                    SELECT AgentName, ActionType, ActionDetails, ExecutionStatus, 
                           HumanApproval, ActionTimestamp
                    FROM AgentActions
                    ORDER BY ActionTimestamp DESC
                """
                return pd.read_sql(query, conn)
            else:
                query = """
                    SELECT AgentName, ActionType, ActionDetails, ExecutionStatus, 
                           HumanApproval, ActionTimestamp
                    FROM AgentActions
                    WHERE DatasetID = ?
                    ORDER BY ActionTimestamp DESC
                """
                return pd.read_sql(query, conn, params=[dataset_id])
        finally:
            conn.close()
