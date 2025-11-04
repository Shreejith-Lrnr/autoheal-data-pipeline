import pyodbc
import pandas as pd
import json
import requests
import io
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
    
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        finally:
            conn.close()
    def ensure_tables_exist(self):
        """Create necessary tables if they don't exist"""
        
        # First, handle table migrations/updates - drop old schema if it exists
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Force drop APISources and APISnapshots if they have old schema
            drop_sql = """
            IF OBJECT_ID('APISnapshots', 'U') IS NOT NULL
                DROP TABLE APISnapshots;
            
            IF OBJECT_ID('APISources', 'U') IS NOT NULL
                DROP TABLE APISources;
            """
            cursor.execute(drop_sql)
            conn.commit()
            cursor.close()
        except Exception as e:
            pass  # Table doesn't exist or already dropped
        finally:
            conn.close()
        
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
            ActionTimestamp DATETIME DEFAULT GETDATE(),
            Success BIT DEFAULT 1,
            ErrorMessage NTEXT,
            Context NVARCHAR(255)
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
        
        -- Agent learning patterns table
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AgentLearningPatterns' AND xtype='U')
        CREATE TABLE AgentLearningPatterns (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            AgentName NVARCHAR(100),
            PatternType NVARCHAR(100),
            Pattern NTEXT,
            SuccessRate DECIMAL(5,2),
            ConfidenceLevel DECIMAL(5,2),
            LastUpdated DATETIME DEFAULT GETDATE(),
            UsageCount INT DEFAULT 0
        );
        
        -- Human feedback table
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='HumanFeedback' AND xtype='U')
        CREATE TABLE HumanFeedback (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            AgentName NVARCHAR(100),
            FeedbackText NTEXT,
            Rating INT,
            Context NVARCHAR(255),
            ActionID INT,
            FeedbackTimestamp DATETIME DEFAULT GETDATE(),
            FOREIGN KEY (ActionID) REFERENCES AgentActions(ID)
        );

        -- API sources table (dynamic live data connectors)
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='APISources' AND xtype='U')
        CREATE TABLE APISources (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            SourceName NVARCHAR(255),
            ApiUrl NVARCHAR(2000),
            FetchInterval INT DEFAULT 300, -- seconds
            IsActive BIT DEFAULT 1,
            LastFetchTimestamp DATETIME NULL,
            CreatedAt DATETIME DEFAULT GETDATE()
        );

        -- API snapshots table (store fetched payloads for audit and processing)
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='APISnapshots' AND xtype='U')
        CREATE TABLE APISnapshots (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            SourceID INT FOREIGN KEY REFERENCES APISources(ID),
            SnapshotData NTEXT,
            SnapshotFormat NVARCHAR(20),
            RecordsCount INT,
            FetchedAt DATETIME DEFAULT GETDATE()
        );
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(create_tables_sql)
            conn.commit()
            
            # Add missing columns to existing tables
            self._update_schema_if_needed(cursor)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Table creation warning: {e}")
    
    def _update_schema_if_needed(self, cursor):
        """Add new columns to existing tables if they don't exist"""
        schema_updates = [
            # Add learning-related columns to AgentActions if they don't exist
            """
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                          WHERE TABLE_NAME = 'AgentActions' AND COLUMN_NAME = 'Success')
            ALTER TABLE AgentActions ADD Success BIT DEFAULT 1
            """,
            """
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                          WHERE TABLE_NAME = 'AgentActions' AND COLUMN_NAME = 'ErrorMessage')
            ALTER TABLE AgentActions ADD ErrorMessage NTEXT
            """,
            """
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                          WHERE TABLE_NAME = 'AgentActions' AND COLUMN_NAME = 'Context')
            ALTER TABLE AgentActions ADD Context NVARCHAR(255)
            """,
            # Add additional columns to ProcessedData for better tracking
            """
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                          WHERE TABLE_NAME = 'ProcessedData' AND COLUMN_NAME = 'ProcessedSize')
            ALTER TABLE ProcessedData ADD ProcessedSize INT
            """
        ]
        
        for update_sql in schema_updates:
            try:
                cursor.execute(update_sql)
            except Exception as e:
                print(f"Schema update warning: {e}")
    
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
    
    # Download-related methods
    def get_processed_datasets(self):
        """Get list of all processed datasets for download selection"""
        conn = self.get_connection()
        try:
            query = """
                SELECT ID, DatasetName, DataQualityScore, ProcessingTimestamp, Status,
                       LEN(OriginalData) as OriginalSize,
                       LEN(ProcessedData) as ProcessedSize
                FROM ProcessedData
                ORDER BY ProcessingTimestamp DESC
            """
            return pd.read_sql(query, conn)
        finally:
            conn.close()
    
    def get_processed_data(self, dataset_id):
        """Get the cleaned/processed data for a specific dataset"""
        conn = self.get_connection()
        try:
            query = "SELECT ProcessedData FROM ProcessedData WHERE ID = ?"
            result = self.execute_query(query, [dataset_id])
            
            if result and len(result) > 0:
                processed_data_json = result[0][0]
                if processed_data_json:
                    # Convert JSON back to DataFrame
                    data_dict = json.loads(processed_data_json)
                    return pd.DataFrame(data_dict)
            return None
        except Exception as e:
            print(f"Error retrieving processed data: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_original_data(self, dataset_id):
        """Get the original data for a specific dataset"""
        conn = self.get_connection()
        try:
            query = "SELECT OriginalData FROM ProcessedData WHERE ID = ?"
            result = self.execute_query(query, [dataset_id])
            
            if result and len(result) > 0:
                original_data_json = result[0][0]
                if original_data_json:
                    data_dict = json.loads(original_data_json)
                    return pd.DataFrame(data_dict)
            return None
        except Exception as e:
            print(f"Error retrieving original data: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_dataset_info(self, dataset_id):
        """Get detailed information about a dataset"""
        conn = self.get_connection()
        try:
            query = """
                SELECT ID, DatasetName, DataQualityScore, ProcessingTimestamp, Status,
                       LEN(OriginalData) as OriginalRecords,
                       LEN(ProcessedData) as FinalRecords
                FROM ProcessedData 
                WHERE ID = ?
            """
            result = self.execute_query(query, [dataset_id])
            
            if result and len(result) > 0:
                return {
                    'ID': result[0][0],
                    'DatasetName': result[0][1],
                    'DataQualityScore': result[0][2],
                    'ProcessingTimestamp': result[0][3],
                    'Status': result[0][4],
                    'OriginalRecords': result[0][5],
                    'FinalRecords': result[0][6]
                }
            return None
        finally:
            conn.close()
    
    def get_quality_issues(self, dataset_id):
        """Get all quality issues for a specific dataset"""
        conn = self.get_connection()
        try:
            query = """
                SELECT IssueType, IssueDescription, AffectedRows, Severity, DetectedTimestamp
                FROM DataQualityIssues
                WHERE DatasetID = ?
                ORDER BY DetectedTimestamp DESC
            """
            return pd.read_sql(query, conn, params=[dataset_id])
        finally:
            conn.close()
    
    def generate_processing_report(self, dataset_id):
        """Generate comprehensive processing report for a dataset"""
        try:
            dataset_info = self.get_dataset_info(dataset_id)
            quality_issues = self.get_quality_issues(dataset_id)
            agent_actions = self.get_agent_actions(dataset_id)
            
            report = {
                'dataset_id': dataset_id,
                'dataset_info': dataset_info,
                'processing_summary': {
                    'quality_score': dataset_info.get('DataQualityScore', 0) if dataset_info else 0,
                    'processing_status': dataset_info.get('Status', 'Unknown') if dataset_info else 'Unknown',
                    'processing_timestamp': dataset_info.get('ProcessingTimestamp').isoformat() if dataset_info and dataset_info.get('ProcessingTimestamp') else None
                },
                'quality_analysis': {
                    'total_issues': len(quality_issues) if not quality_issues.empty else 0,
                    'issues_by_type': quality_issues['IssueType'].value_counts().to_dict() if not quality_issues.empty else {},
                    'issues_by_severity': quality_issues['Severity'].value_counts().to_dict() if not quality_issues.empty else {},
                    'detailed_issues': quality_issues.to_dict('records') if not quality_issues.empty else []
                },
                'agent_actions': {
                    'total_actions': len(agent_actions) if not agent_actions.empty else 0,
                    'actions_by_agent': agent_actions['AgentName'].value_counts().to_dict() if not agent_actions.empty else {},
                    'actions_by_status': agent_actions['ExecutionStatus'].value_counts().to_dict() if not agent_actions.empty else {},
                    'detailed_actions': agent_actions.to_dict('records') if not agent_actions.empty else []
                },
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating processing report: {str(e)}")
            return {
                'error': str(e),
                'dataset_id': dataset_id,
                'generated_at': datetime.now().isoformat()
            }

    # --- Dynamic API source helpers ---
    def register_api_source(self, source_name, api_url, fetch_interval=300):
        """Register a live API source for scheduled fetching"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO APISources (SourceName, ApiUrl, FetchInterval, IsActive)
                VALUES (?, ?, ?, 1)
            """, source_name, api_url, fetch_interval)
            conn.commit()
            return cursor.execute("SELECT @@IDENTITY").fetchone()[0]
        finally:
            conn.close()

    def list_api_sources(self):
        """Return registered API sources"""
        conn = self.get_connection()
        try:
            return pd.read_sql("SELECT ID, SourceName, ApiUrl, FetchInterval, IsActive, LastFetchTimestamp, CreatedAt FROM APISources ORDER BY CreatedAt DESC", conn)
        finally:
            conn.close()

    def store_api_snapshot(self, source_id, snapshot_json, snapshot_format='json', records_count=None):
        """Persist an API snapshot payload for audit and processing"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO APISnapshots (SourceID, SnapshotData, SnapshotFormat, RecordsCount)
                VALUES (?, ?, ?, ?)
            """, source_id, json.dumps(snapshot_json), snapshot_format, records_count)
            # update last fetched timestamp on source
            cursor.execute("UPDATE APISources SET LastFetchTimestamp = ? WHERE ID = ?", datetime.now(), source_id)
            conn.commit()
            return cursor.execute("SELECT @@IDENTITY").fetchone()[0]
        finally:
            conn.close()

    def get_latest_snapshot_for_source(self, source_id):
        """Return the most recent snapshot for a given source as a DataFrame (if possible)"""
        conn = self.get_connection()
        try:
            query = "SELECT TOP 1 SnapshotData, SnapshotFormat FROM APISnapshots WHERE SourceID = ? ORDER BY FetchedAt DESC"
            cursor = conn.cursor()
            cursor.execute(query, source_id)
            row = cursor.fetchone()
            if not row:
                return None
            snapshot_data = row[0]
            snapshot_format = row[1]
            if snapshot_format.lower() == 'json':
                obj = json.loads(snapshot_data)
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    # return raw JSON if it isn't tabular
                    return obj
            elif snapshot_format.lower() == 'csv':
                try:
                    return pd.read_csv(io.StringIO(snapshot_data))
                except Exception:
                    return snapshot_data
            else:
                return snapshot_data
        finally:
            conn.close()

    def fetch_and_store_api(self, source_id=None, url=None, method='GET', data_format='json', headers=None, params=None, timeout=20):
        """Fetch data from a live API endpoint and store a snapshot. If source_id provided, it will update that source's last fetched timestamp."""
        # Use provided URL or lookup by source_id
        if source_id and (url is None):
            conn = self.get_connection()
            try:
                query = "SELECT URL, Method, DataFormat, Headers, Params FROM APISources WHERE ID = ?"
                res = self.execute_query(query, (source_id,))
                if res and len(res) > 0:
                    url = res[0][0]
                    method = res[0][1] or method
                    data_format = res[0][2] or data_format
                    headers = json.loads(res[0][3]) if res[0][3] else headers
                    params = json.loads(res[0][4]) if res[0][4] else params
                else:
                    raise Exception("API source not found")
            finally:
                conn.close()

        # perform HTTP request
        try:
            if method.upper() == 'GET':
                resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            else:
                resp = requests.post(url, headers=headers, json=params, timeout=timeout)

            resp.raise_for_status()

            if data_format.lower() == 'json':
                payload = resp.json()
                # attempt to normalize to tabular if possible
                try:
                    df = pd.json_normalize(payload)
                    records_count = len(df)
                    stored_id = None
                    if source_id:
                        stored_id = self.store_api_snapshot(source_id, payload, 'json', records_count)
                    return df if df is not None else payload
                except Exception:
                    # store raw JSON
                    if source_id:
                        stored_id = self.store_api_snapshot(source_id, payload, 'json', None)
                    return payload
            elif data_format.lower() == 'csv':
                text = resp.text
                df = pd.read_csv(io.StringIO(text))
                if source_id:
                    self.store_api_snapshot(source_id, text, 'csv', len(df))
                return df
            else:
                # unknown format - store raw text
                text = resp.text
                if source_id:
                    self.store_api_snapshot(source_id, text, 'text', None)
                return text

        except Exception as e:
            print(f"Error fetching API ({url}): {str(e)}")
            raise
