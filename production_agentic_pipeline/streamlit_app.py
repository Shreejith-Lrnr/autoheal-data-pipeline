import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import time

from config import Config
from database import ProductionDatabase
from agents import PipelineOrchestratorAgent, DataHealingAgent
from real_time_monitor import get_monitor

# --- Robust file loader for uploads ---
def load_uploaded_file(uploaded_file):
    """Robust file loader that handles different formats"""
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        elif file_extension in ['xlsx', 'xlsm']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(uploaded_file, engine='xlrd')
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        # Basic validation
        if df.empty:
            raise ValueError("File appears to be empty")
        if len(df.columns) == 0:
            raise ValueError("No columns found in file")
        return df, None
    except Exception as e:
        return None, str(e)

# Page configuration
st.set_page_config(
    page_title="AutoHeal Data Pipeline",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    Config.validate_config()
    return ProductionDatabase(), PipelineOrchestratorAgent(), DataHealingAgent(), get_monitor()

db, orchestrator, healing_agent, rt_monitor = init_components()

# Real-time notifications
def show_real_time_notifications():
    """Show real-time notifications in the sidebar"""
    if rt_monitor.get_status()['running']:
        metrics = rt_monitor.get_system_metrics()
        
        # Quality alerts
        if metrics['quality_alerts'] > 0:
            st.sidebar.warning(f"⚠️ {metrics['quality_alerts']} Quality Alert(s)")
        
        # Active jobs notification
        if metrics['active_jobs'] > 0:
            st.sidebar.info(f"🔄 {metrics['active_jobs']} Job(s) Processing")

# Show notifications
show_real_time_notifications()

# Sidebar
st.sidebar.title("🩺 AutoHeal Data Pipeline")
st.sidebar.markdown("**Automated Monitoring & Self-Healing for Data**")

page = st.sidebar.selectbox(
    "Navigate", 
    ["📥 Upload & Auto-Heal", "📈 Monitoring Dashboard", "⚡ Real-Time Dashboard", "📝 Action Log", "🛡️ System Health"]
)

# Real-Time Monitor Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ Real-Time Monitor**")

monitor_status = rt_monitor.get_status()
if monitor_status['running']:
    st.sidebar.success("🟢 Monitor Active")
    if st.sidebar.button("⏹️ Stop Monitor"):
        rt_monitor.stop_monitoring()
        st.sidebar.success("Monitor stopped")
        st.rerun()
else:
    st.sidebar.error("🔴 Monitor Stopped")
    if st.sidebar.button("🚀 Start Monitor"):
        rt_monitor.start_monitoring()
        st.sidebar.success("Monitor started!")
        st.rerun()

# Quick metrics in sidebar
if monitor_status['running']:
    metrics = rt_monitor.get_system_metrics()
    st.sidebar.metric("Active Jobs", metrics['active_jobs'])
    st.sidebar.metric("Today's Quality", f"{metrics['avg_quality_today']:.1f}%")

if page == "📥 Upload & Auto-Heal":
    st.title("🩺 AutoHeal Data Pipeline: Upload & Self-Heal")
    
    # Add refresh button at the top
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("""
        Welcome to **AutoHeal** – the automated, self-healing data pipeline.
        
        **How it works:**
        1. Upload your dataset (CSV, Excel, or JSON)
        2. The pipeline automatically detects and reports data quality issues
        3. AI proposes fixes (self-healing actions)
        4. You approve or reject each fix
        5. All actions and results are monitored and logged
        
        **Key Features:**
        - Automated data quality monitoring
        - AI-powered self-healing suggestions
        - Human-in-the-loop approvals
        - Full audit trail and monitoring
        """)
    
    with col2:
        st.write("")  # Add some spacing
        st.write("")
        
        # Show session status
        if any(key in st.session_state for key in ['uploaded_data', 'sample_data', 'processing_result']):
            st.info("📊 **Session Active**\nData loaded")
            
            # Show what data is currently loaded
            if 'processing_result' in st.session_state:
                st.caption("🩺 Processing complete")
            elif 'uploaded_data' in st.session_state:
                st.caption(f"📁 File: {st.session_state.get('file_dataset_name', 'Unknown')}")
            elif 'sample_data' in st.session_state:
                st.caption(f"🧪 Sample: {st.session_state.get('dataset_name', 'Unknown')}")
        else:
            st.success("✨ **Clean Session**\nReady for new data")
        
        # Two-step refresh process for safety
        if 'confirm_refresh' not in st.session_state:
            st.session_state.confirm_refresh = False
        
        if not st.session_state.confirm_refresh:
            if st.button("🔄 Refresh", help="Clear all data and start fresh", type="secondary"):
                if any(key in st.session_state for key in ['uploaded_data', 'sample_data', 'processing_result']):
                    st.session_state.confirm_refresh = True
                    st.rerun()
                else:
                    st.info("ℹ️ No data to clear - session already clean")
        else:
            st.warning("⚠️ **Confirm Refresh**\nThis will clear all loaded data and processing results")
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ Confirm", type="primary"):
                    # Clear all session state related to data processing
                    keys_to_clear = [
                        'uploaded_data', 'sample_data', 'file_dataset_name', 'dataset_name',
                        'processing_result', 'current_dataset_id', 'current_df', 
                        'execution_logs', 'confirm_refresh'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.success("✅ Session cleared! You can now upload fresh data.")
                    st.rerun()
            
            with col_confirm2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_refresh = False
                    st.rerun()
    
    # Data Upload Section
    st.subheader("📥 Upload Your Data File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files up to 100MB"
        )
        
        dataset_name = st.text_input(
            "Dataset Name", 
            placeholder="e.g., Sales_Q3_2025, Customer_Database, etc."
        )
    
    with col2:
        st.markdown("**Or try a sample dataset:**")
        if st.button("🧪 Load Sales Sample"):
            # Generate sample sales data with intentional issues
            sample_data = pd.DataFrame({
                'OrderID': range(1, 501),
                'CustomerID': [f'CUST_{i:04d}' for i in range(1, 501)],
                'ProductID': [f'PROD_{i%20:03d}' for i in range(1, 501)],
                'OrderDate': pd.date_range('2025-01-01', periods=500, freq='D'),
                'Quantity': [None if i % 25 == 0 else (0 if i % 15 == 0 else i % 50 + 1) for i in range(500)],  # 4% NULL, some zeros
                'UnitPrice': [round(abs(i * 0.99 + 10), 2) if i % 30 != 0 else -round(abs(i * 0.99 + 10), 2) for i in range(500)],  # Some negative values
                'Region': [None if i % 20 == 0 else ['North', 'South', 'East', 'West'][i % 4] for i in range(500)]  # 5% NULL values
            })
            
            # Add some duplicate rows to trigger duplicate detection
            sample_data = pd.concat([sample_data, sample_data.head(10)], ignore_index=True)
            
            st.session_state.sample_data = sample_data
            st.session_state.dataset_name = "Sales_Sample_Data_With_Issues"
            st.success("✅ Sample data loaded with intentional quality issues! Click 'Auto-Heal Data' below to see the pipeline in action.")
            st.info(f"📊 Sample contains: {len(sample_data)} records, {sample_data.isnull().sum().sum()} missing values, {sample_data.duplicated().sum()} duplicates")
    
    # Process uploaded file or sample data
    if uploaded_file is not None:
        with st.spinner("📁 Loading your data..."):
            df, error = load_uploaded_file(uploaded_file)
            if df is not None:
                st.session_state.uploaded_data = df
                st.session_state.file_dataset_name = dataset_name if dataset_name else uploaded_file.name.split('.')[0]
                st.success(f"""
                ✅ **File loaded successfully!**
                - **Rows:** {len(df):,}
                - **Columns:** {len(df.columns)}
                - **File:** {uploaded_file.name}
                - **Size:** {uploaded_file.size / 1024:.1f} KB
                """)
            else:
                st.error(f"❌ **Failed to load file:** {error}")
                with st.expander("🛠️ Need Help?"):
                    st.markdown(f"""
                    **File:** `{uploaded_file.name}`
                    **Type:** `{uploaded_file.type}`
                    **Size:** `{uploaded_file.size / 1024:.1f} KB`
                    
                    **Common Solutions:**
                    - **For Excel files:** Make sure the file isn't password protected
                    - **For CSV files:** Try saving with UTF-8 encoding
                    - **Large files:** Break into smaller chunks if over 50MB
                    - **Corrupted files:** Try opening in Excel/LibreOffice first
                    
                    **Supported Formats:** CSV, XLSX, XLS, JSON
                    """)
    
    # Display data preview
    if 'uploaded_data' in st.session_state or 'sample_data' in st.session_state:
        df = st.session_state.get('uploaded_data', st.session_state.get('sample_data'))
        current_dataset_name = st.session_state.get('file_dataset_name', st.session_state.get('dataset_name', 'Unknown'))
        
        st.subheader("🔎 Data Preview & Quality Snapshot")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        # Show data sample
        st.dataframe(df.head(10), use_container_width=True)
        
        # Process Dataset Button
        if st.button("🩺 Auto-Heal Data", type="primary"):
            with st.spinner("🩺 Auto-healing in progress..."):
                try:
                    processing_result, dataset_id = orchestrator.process_dataset(df, current_dataset_name)
                    
                    st.session_state.processing_result = processing_result
                    st.session_state.current_dataset_id = dataset_id
                    st.session_state.current_df = df
                    
                    st.success("✅ Auto-healing analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    # Display Processing Results
    if 'processing_result' in st.session_state:
        result = st.session_state.processing_result
        
        st.subheader("🩺 Self-Healing & Monitoring Results")
        
        # Executive Summary
        if 'ai_executive_summary' in result:
            st.markdown("### 📈 Pipeline Executive Summary")
            st.info(result['ai_executive_summary'])
        
        # Quality Score
        quality_score = result['quality_report']['quality_score']
        col1, col2 = st.columns([1, 2])
        with col1:
            # Quality score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = quality_score,
                title = {'text': "Health Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Issues breakdown
            if result['quality_report']['issues']:
                issues_df = pd.DataFrame(result['quality_report']['issues'])
                
                fig_bar = px.bar(
                    issues_df, 
                    x='type', 
                    y='affected_rows',
                    color='severity',
                    title="Detected Issues by Type"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.success("🎉 No significant data quality issues found! Your data is healthy.")
        
        # Detailed Issues
        if result['quality_report']['issues']:
            st.markdown("### 🩺 Issue Details & Healing Suggestions")
            
            for i, issue in enumerate(result['quality_report']['issues']):
                with st.expander(f"{issue['type'].replace('_', ' ').title()} - {issue['severity']} Priority"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Affected Rows", issue['affected_rows'])
                    with col2:
                        st.metric("Percentage", f"{issue['percentage']}%")
                    with col3:
                        st.metric("Severity", issue['severity'])
                    
                    if 'column' in issue:
                        st.write(f"**Column:** {issue['column']}")
        
        # Human Approval Section
        if result['human_approvals_needed']:
            st.markdown("### 👤 Approve Self-Healing Actions")
            st.warning("Review and approve the following automated healing actions:")
            
            for i, approval in enumerate(result['human_approvals_needed']):
                with st.expander(f"Approval Required: {approval['action_type'].replace('_', ' ').title()}"):
                    st.write(f"**Impact:** {approval['impact']}")
                    if approval['recommendation']:
                        st.write(f"**AI Recommendation:** {approval['recommendation']['description']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Approve", key=f"approve_{i}"):
                            # Log approval decision
                            approval_action_id = db.log_agent_action(
                                st.session_state.current_dataset_id,
                                "HumanOperator",
                                f"APPROVED_{approval['action_type']}",
                                f"Human approved {approval['action_type']} - {approval['impact']}",
                                "EXECUTING"
                            )
                            
                            # Execute the healing action
                            action_to_execute = result['healing_plan']['actions'][i]
                            recommended_method = approval['recommendation']['method']
                            
                            with st.spinner(f"Executing {approval['action_type']}..."):
                                cleaned_df, exec_log = healing_agent.execute_healing_action(
                                    st.session_state.current_df, 
                                    action_to_execute, 
                                    recommended_method
                                )
                                
                                if exec_log['success']:
                                    st.session_state.current_df = cleaned_df
                                    st.session_state.execution_logs = st.session_state.get('execution_logs', [])
                                    st.session_state.execution_logs.append(exec_log)
                                    
                                    # Recalculate quality score after healing
                                    new_quality_score = healing_agent.recalculate_quality_score(cleaned_df)
                                    
                                    # Update approval action with success
                                    db.update_agent_action(approval_action_id, "COMPLETED", "APPROVED")
                                    
                                    # Store the cleaned data as processed data with new quality score
                                    db.update_dataset_quality_score(
                                        st.session_state.current_dataset_id, 
                                        new_quality_score, 
                                        cleaned_df, 
                                        "HEALING_IN_PROGRESS"
                                    )
                                    
                                    # Log detailed execution results
                                    db.log_agent_action(
                                        st.session_state.current_dataset_id,
                                        "DataHealingAgent",
                                        f"EXECUTED_{action_to_execute['type']}",
                                        f"Successfully executed {recommended_method} - Original: {exec_log['original_shape']}, Final: {exec_log['final_shape']}, Records affected: {exec_log.get('records_affected', 0)}, New Quality Score: {new_quality_score:.1f}/100",
                                        "COMPLETED"
                                    )
                                    
                                    st.success(f"✅ {approval['action_type']} completed successfully! Quality score updated to {new_quality_score:.1f}/100")
                                    st.rerun()
                                else:
                                    # Update approval action with failure
                                    db.update_agent_action(approval_action_id, "FAILED", "APPROVED")
                                    
                                    # Log failure details
                                    db.log_agent_action(
                                        st.session_state.current_dataset_id,
                                        "DataHealingAgent",
                                        f"FAILED_{action_to_execute['type']}",
                                        f"Execution failed using {recommended_method}: {exec_log['message']}",
                                        "FAILED"
                                    )
                                    
                                    st.error(f"❌ Execution failed: {exec_log['message']}")
                    
                    with col2:
                        if st.button(f"❌ Reject", key=f"reject_{i}"):
                            # Log rejection with detailed reason
                            db.log_agent_action(
                                st.session_state.current_dataset_id,
                                "HumanOperator",
                                f"REJECTED_{approval['action_type']}",
                                f"Human rejected {approval['action_type']} - Impact: {approval['impact']} - Reason: Manual review required",
                                "REJECTED"
                            )
                            
                            # Update the original healing proposal to rejected
                            if 'action_id' in approval:
                                db.update_agent_action(approval['action_id'], "REJECTED", "REJECTED")
                            
                            st.warning(f"❌ Action rejected. Issue logged for manual review.")
                            st.rerun()
        
        # Execution Logs
        if 'execution_logs' in st.session_state:
            st.markdown("### 📝 Healing Action History")
            
            # Check if all healing actions are complete
            all_actions_complete = len(st.session_state.execution_logs) >= len(result.get('human_approvals_needed', []))
            
            if all_actions_complete:
                # Final quality score calculation
                final_quality_score = healing_agent.recalculate_quality_score(st.session_state.current_df)
                
                # Mark dataset as fully completed
                db.update_dataset_quality_score(
                    st.session_state.current_dataset_id,
                    final_quality_score,
                    st.session_state.current_df,
                    "COMPLETED"
                )
                
                # Log final completion
                db.log_agent_action(
                    st.session_state.current_dataset_id,
                    "PipelineOrchestrator",
                    "HEALING_COMPLETE",
                    f"All healing actions completed - Final Quality Score: {final_quality_score:.1f}/100",
                    "COMPLETED"
                )
                
                st.success(f"🎉 All healing actions completed! Final Quality Score: {final_quality_score:.1f}/100")
            
            for log in st.session_state.execution_logs:
                with st.expander(f"✅ {log['action']['type']} - {log['method']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Original Shape:** {log['original_shape']}")
                        st.write(f"**Final Shape:** {log['final_shape']}")
                    with col2:
                        st.write(f"**Records Affected:** {log['records_affected']}")
                        st.write(f"**Status:** {'Success' if log['success'] else 'Failed'}")
                    st.write(f"**Details:** {log['message']}")

elif page == "📈 Monitoring Dashboard":
    st.title("📈 Automated Pipeline Monitoring")
    
    # Auto-refresh toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("")  # Spacing
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", key="monitoring_refresh")
    
    # Get dataset summary
    dataset_summary = db.get_dataset_summary()
    
    if not dataset_summary.empty:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Datasets", len(dataset_summary))
        with col2:
            avg_quality = dataset_summary['DataQualityScore'].mean()
            st.metric("Avg Quality Score", f"{avg_quality:.1f}/100")
        with col3:
            total_issues = dataset_summary['IssuesFound'].sum()
            st.metric("Total Issues Found", total_issues)
        with col4:
            processing_count = len(dataset_summary[dataset_summary['Status'] == 'PROCESSING'])
            st.metric("Currently Processing", processing_count)
        
        # Dataset table
        st.subheader("🗂️ Data Processing History")
        st.dataframe(dataset_summary, use_container_width=True)
        
        # Quality trend
        if len(dataset_summary) > 1:
            fig_trend = px.line(
                dataset_summary.sort_values('ProcessingTimestamp'),
                x='ProcessingTimestamp',
                y='DataQualityScore',
                title="Data Health Trend Over Time"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No datasets processed yet. Upload and process your first dataset!")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

elif page == "⚡ Real-Time Dashboard":
    st.title("⚡ Real-Time Pipeline Dashboard")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write("**Live monitoring of pipeline activity and system health**")
    with col2:
        refresh_interval = st.selectbox("Refresh Rate", [5, 10, 30, 60], index=1, key="rt_refresh_rate")
    with col3:
        auto_refresh_rt = st.checkbox("🔄 Auto-refresh", value=True, key="rt_auto_refresh")
    
    # System Status
    monitor_status = rt_monitor.get_status()
    
    if monitor_status['running']:
        st.success(f"🟢 **Real-Time Monitor Active** - Running for {str(monitor_status['uptime']).split('.')[0] if monitor_status['uptime'] else '0:00:00'}")
    else:
        st.error("🔴 **Real-Time Monitor Stopped** - Start it from the sidebar to see live data")
    
    # Real-time metrics
    metrics = rt_monitor.get_system_metrics()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Datasets Today", metrics['datasets_today'])
    with col2:
        st.metric("Active Jobs", metrics['active_jobs'])
    with col3:
        st.metric("Avg Quality Today", f"{metrics['avg_quality_today']:.1f}%")
    with col4:
        st.metric("Recent Actions", metrics['recent_actions'])
    with col5:
        quality_color = "normal" if metrics['quality_alerts'] == 0 else "inverse"
        st.metric("Quality Alerts", metrics['quality_alerts'], delta=None)
    
    # Live Activity Feed
    st.subheader("📡 Live Activity Feed")
    
    recent_actions = rt_monitor.get_recent_agent_actions(15)
    
    if not recent_actions.empty:
        for _, action in recent_actions.iterrows():
            # Color coding for different statuses
            status_colors = {
                'COMPLETED': '🟢',
                'EXECUTING': '🟡',
                'PENDING_APPROVAL': '🟡',
                'FAILED': '🔴',
                'REJECTED': '🔴',
                'AUTO_APPROVED': '🟢'
            }
            
            status_icon = status_colors.get(action['ExecutionStatus'], '⚫')
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                with col1:
                    st.write(f"**{action['AgentName']}**: {action['ActionType']}")
                with col2:
                    # Truncate long details
                    details = str(action['ActionDetails'])
                    if len(details) > 50:
                        details = details[:50] + "..."
                    st.write(details)
                with col3:
                    st.write(f"{status_icon} {action['ExecutionStatus']}")
                with col4:
                    st.write(action['ActionTimestamp'].strftime("%H:%M:%S"))
                
                st.divider()
    else:
        st.info("No recent actions to display. Upload and process data to see live activity.")
    
    # Monitor Activity Log
    st.subheader("🔍 Monitor Activity Log")
    
    activity_log = rt_monitor.get_recent_activity()
    
    if activity_log:
        # Show last 10 monitor activities
        for activity in activity_log[-10:]:
            timestamp = activity['timestamp'].strftime("%H:%M:%S")
            st.text(f"[{timestamp}] {activity['message']}")
    else:
        st.info("Start the real-time monitor to see activity logs.")
    
    # System Health Indicators
    st.subheader("🛡️ System Health Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Database health
        try:
            test_conn = db.get_connection()
            test_conn.close()
            st.success("🟢 Database: Healthy")
        except:
            st.error("🔴 Database: Connection Issues")
    
    with col2:
        # AI API health
        if metrics['recent_actions'] > 0:
            st.success("🟢 AI Pipeline: Active")
        else:
            st.warning("🟡 AI Pipeline: Idle")
    
    with col3:
        # Data freshness
        if metrics['datasets_today'] > 0:
            st.success(f"🟢 Data: Fresh ({metrics['datasets_today']} today)")
        else:
            st.warning("🟡 Data: No new datasets today")
    
    # Auto-refresh logic
    if auto_refresh_rt and monitor_status['running']:
        time.sleep(refresh_interval)
        st.rerun()
    elif auto_refresh_rt and not monitor_status['running']:
        st.warning("⚠️ Auto-refresh disabled: Start the monitor first")

elif page == "📝 Action Log":
    st.title("📝 Self-Healing & Action Log")
    
    # Auto-refresh toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("")  # Spacing
    with col2:
        auto_refresh_log = st.checkbox("🔄 Auto-refresh (15s)", key="action_log_refresh")
    
    # Debug section (can be removed in production)
    with st.expander("🔍 Debug: Database Status"):
        if st.button("Check Database Tables"):
            try:
                conn = db.get_connection()
                cursor = conn.cursor()
                
                # Check ProcessedData table
                cursor.execute("SELECT COUNT(*) FROM ProcessedData")
                processed_count = cursor.fetchone()[0]
                st.write(f"📊 ProcessedData records: **{processed_count}**")
                
                # Check AgentActions table  
                cursor.execute("SELECT COUNT(*) FROM AgentActions")
                actions_count = cursor.fetchone()[0]
                st.write(f"🤖 AgentActions records: **{actions_count}**")
                
                if actions_count > 0:
                    cursor.execute("SELECT TOP 5 AgentName, ActionType, ExecutionStatus, ActionTimestamp FROM AgentActions ORDER BY ActionTimestamp DESC")
                    recent_actions = cursor.fetchall()
                    st.write("**Recent actions:**")
                    for action in recent_actions:
                        st.write(f"• **{action[0]}**: {action[1]} ({action[2]}) - {action[3]}")
                else:
                    st.warning("No actions found in database. Try uploading data and clicking 'Auto-Heal Data' first.")
                
                # Check DataQualityIssues table
                cursor.execute("SELECT COUNT(*) FROM DataQualityIssues")
                issues_count = cursor.fetchone()[0]
                st.write(f"⚠️ DataQualityIssues records: **{issues_count}**")
                
                conn.close()
                
            except Exception as e:
                st.error(f"Database check failed: {e}")
        
        st.info("💡 **Tip**: Action Log entries are created when you upload data and click 'Auto-Heal Data' on the Upload page.")
    
    # Get agent actions
    agent_actions = db.get_agent_actions()
    
    if not agent_actions.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Actions", len(agent_actions))
        with col2:
            completed_actions = len(agent_actions[agent_actions['ExecutionStatus'] == 'COMPLETED'])
            st.metric("Completed", completed_actions)
        with col3:
            pending_actions = len(agent_actions[agent_actions['ExecutionStatus'].isin(['PENDING_APPROVAL', 'EXECUTING'])])
            st.metric("Pending", pending_actions)
        with col4:
            unique_agents = agent_actions['AgentName'].nunique()
            st.metric("Active Agents", unique_agents)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_agent = st.selectbox(
                "Filter by Agent", 
                ["All"] + list(agent_actions['AgentName'].unique())
            )
        with col2:
            selected_status = st.selectbox(
                "Filter by Status",
                ["All"] + list(agent_actions['ExecutionStatus'].unique())
            )
        
        # Apply filters
        filtered_actions = agent_actions.copy()
        if selected_agent != "All":
            filtered_actions = filtered_actions[filtered_actions['AgentName'] == selected_agent]
        if selected_status != "All":
            filtered_actions = filtered_actions[filtered_actions['ExecutionStatus'] == selected_status]
        
        st.subheader(f"📋 Actions ({len(filtered_actions)} of {len(agent_actions)})")
        
        # Display actions with better formatting
        for _, action in filtered_actions.iterrows():
            # Color coding based on status
            status_color = {
                'COMPLETED': '🟢',
                'EXECUTING': '🟡', 
                'PENDING_APPROVAL': '🟡',
                'FAILED': '🔴',
                'REJECTED': '🔴',
                'AUTO_APPROVED': '🟢'
            }.get(action['ExecutionStatus'], '⚫')
            
            with st.expander(f"{status_color} {action['AgentName']} - {action['ActionType']} ({action['ActionTimestamp']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Status:** {action['ExecutionStatus']}")
                    st.write(f"**Human Approval:** {action['HumanApproval'] or 'N/A'}")
                with col2:
                    st.write(f"**Timestamp:** {action['ActionTimestamp']}")
                    st.write(f"**Agent:** {action['AgentName']}")
                
                st.write("**Action Details:**")
                st.info(action['ActionDetails'])
    else:
        st.info("📭 No agent actions logged yet.")
        st.markdown("""
        **To generate action log entries:**
        1. Go to **📥 Upload & Auto-Heal** page
        2. Upload a CSV/Excel file or use the sample data
        3. Click **🩺 Auto-Heal Data** 
        4. Return here to see the action log entries
        
        The system will automatically log all AI agent activities, quality checks, and healing actions.
        """)
    
    # Auto-refresh logic for Action Log
    if auto_refresh_log:
        time.sleep(15)
        st.rerun()

elif page == "🛡️ System Health":
    st.title("🛡️ System Health & Configuration")
    
    # System health indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Database connection test
        try:
            test_conn = db.get_connection()
            test_conn.close()
            st.success("🟢 Database: Connected")
        except:
            st.error("🔴 Database: Connection Failed")
    
    with col2:
        # API connection test  
        try:
            test_response = orchestrator.call_groq_api("test", temperature=0)
            if "offline mode" in test_response.lower():
                st.warning("🟡 AI API: Using Fallback (Connection Issues)")
            else:
                st.success("🟢 AI API: Connected")
        except Exception as e:
            st.error(f"🔴 AI API: Connection Failed - {str(e)[:50]}...")
            st.info("💡 The system will use fallback responses when API is unavailable")
    
    with col3:
        # System status
        st.success("🟢 System: Operational")
    
    # Configuration display
    st.subheader("📋 System Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        config_data = {
            "Database": "MS SQL Server",
            "AI Model": Config.GROQ_MODEL,
            "Batch Size": Config.BATCH_SIZE,
            "Max File Size": f"{Config.MAX_FILE_SIZE_MB}MB",
        }
        
        for key, value in config_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.write("**Supported Formats:**")
        st.write(", ".join(Config.SUPPORTED_FILE_TYPES))
        
        # Model selection
        st.subheader("🤖 AI Model Selection")
        current_model = Config.GROQ_MODEL
        selected_model = st.selectbox(
            "Select AI Model:", 
            Config.AVAILABLE_GROQ_MODELS,
            index=Config.AVAILABLE_GROQ_MODELS.index(current_model) if current_model in Config.AVAILABLE_GROQ_MODELS else 0
        )
        
        if st.button("🔄 Switch Model"):
            Config.GROQ_MODEL = selected_model
            st.success(f"✅ Switched to model: {selected_model}")
            st.info("💡 New model will be used for subsequent AI operations")
            st.rerun()
    
    # Troubleshooting section
    st.subheader("🔧 Troubleshooting")
    
    if st.button("🧪 Test AI Connection"):
        with st.spinner("Testing AI API connection..."):
            try:
                test_response = orchestrator.call_groq_api("Hello, this is a test message. Please respond briefly.", temperature=0)
                if "offline mode" in test_response.lower() or "fallback" in test_response.lower():
                    st.warning("⚠️ **API Connection Issues Detected**")
                    st.write("**Response received:**")
                    st.info(test_response)
                    
                    with st.expander("🛠️ Troubleshooting Steps"):
                        st.markdown("""
                        **Common 400 Bad Request causes:**
                        1. **Invalid API Key** - Check your GROQ_API_KEY in .env file
                        2. **Model not available** - Try switching to a different model above
                        3. **Request format issues** - Check if the current model supports the request format
                        4. **Rate limiting** - Wait a few seconds and try again
                        5. **Account issues** - Verify your Groq account status
                        
                        **Quick fixes:**
                        - Try switching to 'llama3-70b-8192' or 'mixtral-8x7b-32768'
                        - Check if your API key is valid and has credits
                        - Verify your Groq account is active
                        """)
                else:
                    st.success("✅ **AI API Connection Successful!**")
                    st.write("**Response received:**")
                    st.info(test_response)
                    
            except Exception as e:
                st.error(f"❌ **AI API Test Failed:** {str(e)}")
                
                with st.expander("🛠️ Error Details & Solutions"):
                    st.code(str(e), language="text")
                    st.markdown("""
                    **Possible solutions:**
                    1. **Check API Key:** Verify GROQ_API_KEY is correct in your .env file
                    2. **Try Different Model:** Use the model selector above
                    3. **Network Issues:** Check your internet connection
                    4. **Account Status:** Verify your Groq account has available credits
                    5. **Firewall/Proxy:** Corporate networks may block API calls
                    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🩺 AutoHeal Data Pipeline v1.0**")
st.sidebar.markdown("Automated monitoring & self-healing for your data")