import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from database import ProductionDatabase
from agents import PipelineOrchestratorAgent, DataHealingAgent

class RealTimeDataMonitor:
    def __init__(self, check_interval_seconds=30):
        self.db = ProductionDatabase()
        self.orchestrator = PipelineOrchestratorAgent()
        self.healing_agent = DataHealingAgent()
        self.check_interval = check_interval_seconds
        self.running = False
        self.monitor_thread = None
        self.last_activity_check = datetime.now()
        self.activity_log = []
    
    def start_monitoring(self):
        """Start real-time monitoring in background"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self._log_activity("🚀 Real-time monitoring started")
            return True
        return False
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            self._log_activity("⏹️ Real-time monitoring stopped")
            return True
        return False
    
    def get_status(self):
        """Get current monitor status"""
        return {
            'running': self.running,
            'last_check': self.last_activity_check,
            'uptime': datetime.now() - self.last_activity_check if self.running else None
        }
    
    def get_recent_activity(self):
        """Get recent activity log (last 50 items)"""
        return self.activity_log[-50:]
    
    def _log_activity(self, message):
        """Log activity with timestamp"""
        self.activity_log.append({
            'timestamp': datetime.now(),
            'message': message
        })
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.last_activity_check = datetime.now()
                
                # Check and auto-fetch API sources
                api_fetches_count = self._check_and_fetch_apis()
                
                # Check for new unprocessed data
                new_datasets_count = self._check_for_new_data()
                
                # Check for quality alerts
                quality_alerts_count = self._check_quality_alerts()
                
                # Check for stalled processes
                stalled_count = self._check_stalled_processes()
                
                # Log periodic status
                if new_datasets_count > 0 or quality_alerts_count > 0 or stalled_count > 0 or api_fetches_count > 0:
                    self._log_activity(f"📊 Monitor check: {api_fetches_count} API fetches, {new_datasets_count} new datasets, {quality_alerts_count} quality alerts, {stalled_count} stalled processes")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self._log_activity(f"❌ Monitor error: {str(e)}")
                time.sleep(self.check_interval)
    
    def _check_and_fetch_apis(self):
        """Check active API sources and auto-fetch based on intervals"""
        fetches_performed = 0
        try:
            # Get all active API sources
            api_sources = self.db.list_api_sources()
            
            if api_sources.empty:
                return 0
            
            current_time = datetime.now()
            
            for _, source in api_sources.iterrows():
                if not source['IsActive']:
                    continue
                
                source_id = source['ID']
                fetch_interval = source['FetchInterval']  # in seconds
                last_fetch = source['LastFetchTimestamp']
                
                # Skip if interval is 0 (manual fetch only)
                if fetch_interval == 0:
                    continue
                
                # Check if it's time to fetch
                should_fetch = False
                if pd.isna(last_fetch):
                    should_fetch = True  # Never fetched before
                else:
                    time_since_last = (current_time - last_fetch).total_seconds()
                    should_fetch = time_since_last >= fetch_interval
                
                if should_fetch:
                    try:
                        # Perform auto-fetch
                        result = self.db.fetch_and_store_api(
                            source_id=source_id,
                            url=source['ApiUrl'],
                            method='GET',
                            data_format='json',
                            timeout=15
                        )
                        
                        record_count = len(result) if isinstance(result, pd.DataFrame) else 0
                        self._log_activity(f"🔄 Auto-fetched API: {source['SourceName']} ({record_count} records)")
                        fetches_performed += 1
                        
                    except Exception as fetch_error:
                        self._log_activity(f"❌ Auto-fetch failed for {source['SourceName']}: {str(fetch_error)}")
            
            return fetches_performed
            
        except Exception as e:
            self._log_activity(f"❌ Error in API auto-fetch: {str(e)}")
            return 0
    
    def _check_for_new_data(self):
        """Check for new unprocessed datasets"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check for datasets that are still processing or failed
            cursor.execute("""
                SELECT COUNT(*) FROM ProcessedData 
                WHERE Status IN ('PROCESSING', 'FAILED') 
                AND ProcessingTimestamp > DATEADD(minute, -30, GETDATE())
            """)
            
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                self._log_activity(f"🔄 Found {count} datasets requiring attention")
            
            return count
            
        except Exception as e:
            self._log_activity(f"❌ Error checking for new data: {str(e)}")
            return 0
    
    def _check_quality_alerts(self):
        """Check for quality score degradation"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check for recent datasets with low quality scores
            cursor.execute("""
                SELECT DatasetName, DataQualityScore 
                FROM ProcessedData 
                WHERE ProcessingTimestamp > DATEADD(hour, -2, GETDATE())
                AND DataQualityScore < 70
                AND DataQualityScore > 0
            """)
            
            alerts = cursor.fetchall()
            conn.close()
            
            if alerts:
                for dataset_name, score in alerts:
                    self._log_activity(f"⚠️ Quality alert: {dataset_name} scored {score:.1f}% (below 70%)")
            
            return len(alerts)
            
        except Exception as e:
            self._log_activity(f"❌ Error checking quality alerts: {str(e)}")
            return 0
    
    def _check_stalled_processes(self):
        """Check for processes that have been running too long"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check for datasets stuck in PROCESSING status for over 10 minutes
            cursor.execute("""
                SELECT DatasetName, ProcessingTimestamp
                FROM ProcessedData 
                WHERE Status = 'PROCESSING' 
                AND ProcessingTimestamp < DATEADD(minute, -10, GETDATE())
            """)
            
            stalled = cursor.fetchall()
            conn.close()
            
            if stalled:
                for dataset_name, timestamp in stalled:
                    self._log_activity(f"🚨 Stalled process detected: {dataset_name} (started {timestamp})")
            
            return len(stalled)
            
        except Exception as e:
            self._log_activity(f"❌ Error checking stalled processes: {str(e)}")
            return 0
    
    def get_system_metrics(self):
        """Get real-time system metrics"""
        try:
            conn = self.db.get_connection()
            
            metrics = {}
            
            # Total datasets processed today
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM ProcessedData 
                WHERE ProcessingTimestamp >= CAST(GETDATE() AS DATE)
            """)
            metrics['datasets_today'] = cursor.fetchone()[0]
            
            # Active processing jobs
            cursor.execute("""
                SELECT COUNT(*) FROM ProcessedData 
                WHERE Status IN ('PROCESSING', 'PENDING_APPROVAL')
            """)
            metrics['active_jobs'] = cursor.fetchone()[0]
            
            # Average quality score today
            cursor.execute("""
                SELECT ISNULL(AVG(DataQualityScore), 0) FROM ProcessedData 
                WHERE ProcessingTimestamp >= CAST(GETDATE() AS DATE)
                AND DataQualityScore > 0
            """)
            metrics['avg_quality_today'] = round(cursor.fetchone()[0], 1)
            
            # Recent actions (last hour)
            cursor.execute("""
                SELECT COUNT(*) FROM AgentActions 
                WHERE ActionTimestamp >= DATEADD(hour, -1, GETDATE())
            """)
            metrics['recent_actions'] = cursor.fetchone()[0]
            
            # Quality alerts (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) FROM ProcessedData 
                WHERE ProcessingTimestamp >= DATEADD(day, -1, GETDATE())
                AND DataQualityScore < 70 AND DataQualityScore > 0
            """)
            metrics['quality_alerts'] = cursor.fetchone()[0]
            
            conn.close()
            return metrics
            
        except Exception as e:
            self._log_activity(f"❌ Error getting system metrics: {str(e)}")
            return {
                'datasets_today': 0,
                'active_jobs': 0,
                'avg_quality_today': 0.0,
                'recent_actions': 0,
                'quality_alerts': 0
            }
    
    def get_recent_agent_actions(self, limit=10):
        """Get most recent agent actions for live feed"""
        try:
            conn = self.db.get_connection()
            
            query = f"""
                SELECT TOP {limit} AgentName, ActionType, ActionDetails, 
                       ExecutionStatus, ActionTimestamp
                FROM AgentActions 
                ORDER BY ActionTimestamp DESC
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            self._log_activity(f"❌ Error getting recent actions: {str(e)}")
            return pd.DataFrame()

# Global monitor instance
_monitor_instance = None

def get_monitor():
    """Get global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealTimeDataMonitor()
    return _monitor_instance