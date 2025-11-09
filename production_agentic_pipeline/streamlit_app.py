import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import json
import time

from config import Config
from database import ProductionDatabase
from agents import PipelineOrchestratorAgent, DataHealingAgent, DataQualityAgent
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
    db = ProductionDatabase()
    orchestrator = PipelineOrchestratorAgent()
    healing_agent = DataHealingAgent()
    quality_agent = DataQualityAgent()
    rt_monitor = get_monitor()
    
    # Auto-start the real-time monitor for API fetching
    if not rt_monitor.get_status()['running']:
        rt_monitor.start_monitoring()
    
    return db, orchestrator, healing_agent, quality_agent, rt_monitor

db, orchestrator, healing_agent, quality_agent, rt_monitor = init_components()

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
    ["📥 Upload & Process Data", "⚙️ Settings & Health"]
)

# Real-Time Monitor Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ Real-Time Monitor**")
st.sidebar.markdown("*Auto-started for API fetching*")

monitor_status = rt_monitor.get_status()
if monitor_status['running']:
    st.sidebar.success("🟢 Monitor Active")
    st.sidebar.caption("Automatically fetching APIs based on intervals")
    if st.sidebar.button("⏹️ Stop Monitor"):
        rt_monitor.stop_monitoring()
        st.sidebar.success("Monitor stopped")
        st.rerun()
else:
    st.sidebar.error("🔴 Monitor Stopped")
    st.sidebar.caption("API auto-fetching disabled")
    if st.sidebar.button("🚀 Start Monitor"):
        rt_monitor.start_monitoring()
        st.sidebar.success("Monitor started!")
        st.rerun()

# Quick metrics in sidebar
if monitor_status['running']:
    metrics = rt_monitor.get_system_metrics()
    api_stats = rt_monitor.get_api_fetch_stats()
    
    st.sidebar.metric("Active Jobs", metrics['active_jobs'])
    st.sidebar.metric("Today's Quality", f"{metrics['avg_quality_today']:.1f}%")
    st.sidebar.metric("API Sources", f"{api_stats['active_sources']}/{api_stats['total_sources']}")
    st.sidebar.metric("API Fetches Today", api_stats['snapshots_today'])

if page == "📥 Upload & Process Data":
    st.title("🩺 AutoHeal Data Pipeline: Upload & Process")
    
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
        
        # Simple refresh button
        if st.button("🔄 Clear Session", help="Clear all data and start fresh"):
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
            # Create progress tracking containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            with status_container:
                status_text = st.empty()
            
            try:
                # Step 1: Dataset Storage (10%)
                progress_text.text("📦 Storing dataset in database...")
                progress_bar.progress(10)
                status_text.info("🔄 Step 1/6: Dataset Storage - Saving your data securely...")
                
                # Step 2: Quality Analysis (30%)
                progress_text.text("🔍 Analyzing data quality...")
                progress_bar.progress(30)
                status_text.info("🔄 Step 2/6: Quality Analysis - Scanning for issues...")
                
                # Step 3: Healing Plan Generation (50%)
                progress_text.text("🧠 Generating healing plan with AI...")
                progress_bar.progress(50)
                status_text.info("🔄 Step 3/6: AI Planning - Creating intelligent healing strategies...")
                
                # Step 4: Auto-Execution (70%)
                progress_text.text("� Auto-executing high-confidence actions...")
                progress_bar.progress(70)
                status_text.info("🔄 Step 4/6: Auto-Healing - Applying safe, automated fixes...")
                
                # Step 5: Approval Preparation (90%)
                progress_text.text("📋 Preparing actions for approval...")
                progress_bar.progress(90)
                status_text.info("🔄 Step 5/6: Review Preparation - Setting up human oversight...")
                
                # Step 6: Final Summary (100%)
                progress_text.text("📊 Generating executive summary...")
                progress_bar.progress(100)
                status_text.info("🔄 Step 6/6: Final Summary - Creating business impact assessment...")
                
                # Actually run the processing
                processing_result, dataset_id = orchestrator.process_dataset(df, current_dataset_name)
                
                # Clear progress indicators
                progress_container.empty()
                status_container.empty()
                
                st.session_state.processing_result = processing_result
                st.session_state.current_dataset_id = dataset_id
                # Store the dataframe WITH auto-executed changes
                st.session_state.current_df = processing_result.get('current_df', df)
                
                # Show summary of auto-executed actions
                auto_count = len(processing_result.get('auto_executed_actions', []))
                if auto_count > 0:
                    st.success(f"✅ Auto-healing complete! 🤖 {auto_count} action(s) auto-executed autonomously.")
                else:
                    st.success("✅ Auto-healing analysis complete!")
                st.rerun()
                
            except Exception as e:
                # Clear progress indicators on error
                progress_container.empty()
                status_container.empty()
                st.error(f"Processing error: {str(e)}")

    # Display Processing Results
    if 'processing_result' in st.session_state:
        result = st.session_state.processing_result
        
        st.subheader("🩺 Self-Healing & Monitoring Results")
        
        # Executive Summary
        if 'ai_executive_summary' in result:
            st.markdown("### 📈 Pipeline Executive Summary")

            # Function to highlight quality score and issues in the summary
            def highlight_summary_metrics(summary_text, quality_score, issues_count):
                # Highlight quality score with color based on value
                if quality_score >= 80:
                    score_color = "#10b981"  # green-500
                elif quality_score >= 60:
                    score_color = "#f59e0b"  # amber-500
                else:
                    score_color = "#ef4444"  # red-500

                # Highlight issues count
                issues_color = "#f97316"  # orange-500

                # Replace quality score mentions
                import re
                summary_text = re.sub(
                    r'(\d+(?:\.\d+)?)/100',
                    f'<span style="color: {score_color}; font-weight: bold;">\\1/100</span>',
                    summary_text
                )

                # Replace issues count mentions
                summary_text = re.sub(
                    r'(\d+)\s*(?:issues?|problems?)',
                    f'<span style="color: {issues_color}; font-weight: bold;">\\1 issues</span>',
                    summary_text,
                    flags=re.IGNORECASE
                )

                return summary_text

            highlighted_summary = highlight_summary_metrics(
                result['ai_executive_summary'],
                result['quality_report']['quality_score'],
                len(result['quality_report']['issues'])
            )

            st.markdown(
                f'<div style="background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 4px;">{highlighted_summary}</div>',
                unsafe_allow_html=True
            )
        
        # ========== AUTO-EXECUTED ACTIONS SECTION ==========
        if result.get('auto_executed_actions'):
            st.markdown("### ✅ Auto-Executed Actions (High Confidence)")
            st.success(f"🤖 Agent autonomously executed {len(result['auto_executed_actions'])} high-confidence healing actions")
            
            for i, auto_action in enumerate(result['auto_executed_actions']):
                action = auto_action['action']
                exec_log = auto_action['execution_log']
                confidence = auto_action['confidence']
                
                with st.expander(f"✓ {action['type'].replace('_', ' ').title()} - {confidence:.0%} Confidence (Auto-Executed)", expanded=False):
                    # Show confidence and risk
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Confidence", f"{confidence:.0%}")
                    with col2:
                        st.metric("Risk Level", action.get('risk_level', 'LOW'))
                    with col3:
                        st.metric("Affected Rows", action.get('affected_rows', 0))
                    with col4:
                        status_icon = "✅" if exec_log['success'] else "❌"
                        st.metric("Status", f"{status_icon} {'Success' if exec_log['success'] else 'Failed'}")
                    
                    # Show reasoning (chain of thought)
                    if 'reasoning' in action:
                        st.markdown("**🧠 Agent Reasoning:**")
                        reasoning = action['reasoning']
                        
                        st.write("**Thought Process:**")
                        for thought in reasoning.get('thought_process', []):
                            st.write(f"• {thought}")
                        
                        if reasoning.get('alternatives'):
                            st.write("**Alternatives Considered:**")
                            for alt in reasoning['alternatives']:
                                st.write(f"  - {alt}")
                        
                        if reasoning.get('risks'):
                            st.warning("**Risks Identified:**")
                            for risk in reasoning['risks']:
                                st.write(f"  ⚠️ {risk}")
                    
                    # Show execution details
                    st.markdown("**📊 Execution Details:**")
                    st.write(f"**Method Used:** {action.get('recommended_method', 'N/A').replace('_', ' ').title()}")
                    st.write(f"**Column:** {action.get('column', 'Multiple')}")
                    st.write(f"**Original Shape:** {exec_log.get('original_shape', 'N/A')}")
                    st.write(f"**Final Shape:** {exec_log.get('final_shape', 'N/A')}")
                    st.write(f"**Records Modified:** {exec_log.get('records_affected', 0)}")
                    st.info(f"ℹ️ {exec_log.get('message', 'Action executed successfully')}")
        
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
        
        # Human Approval Section - Bulk Approval System
        if result['human_approvals_needed']:
            st.markdown("### 👤 Bulk Approve Self-Healing Actions")
            st.warning("Review and approve multiple automated healing actions at once:")

            # Bulk Selection Controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                select_all = st.checkbox("Select All Actions", key="select_all_approvals")
                
                # Handle select all logic immediately
                if select_all:
                    all_indices = list(range(len(result['human_approvals_needed'])))
                    if st.session_state.get('selected_approvals', []) != all_indices:
                        st.session_state.selected_approvals = all_indices
                        st.rerun()
                elif st.session_state.get('selected_approvals', []) == list(range(len(result['human_approvals_needed']))):
                    # If all are selected but checkbox is unchecked, clear selections
                    st.session_state.selected_approvals = []
                    st.rerun()
            with col2:
                if st.button("✅ Approve Selected", type="primary", disabled=not st.session_state.get('selected_approvals', [])):
                    selected_indices = st.session_state.get('selected_approvals', [])
                    if selected_indices:
                        # ========== OPTIMIZED BATCH PROCESSING ==========
                        
                        # Show progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Execute ALL healing actions first (batch processing)
                        execution_results = []
                        current_df = st.session_state.current_df.copy()
                        
                        status_text.text(f"⚙️ Processing {len(selected_indices)} healing actions...")
                        
                        for idx, i in enumerate(selected_indices):
                            if i < len(result['human_approvals_needed']):
                                approval = result['human_approvals_needed'][i]
                                action_to_execute = result['healing_plan']['actions'][i]
                                recommended_method = approval['recommendation']['method'] if approval.get('recommendation') else 'default'
                                
                                try:
                                    # Execute healing (no DB writes yet)
                                    cleaned_df, exec_log = healing_agent.execute_healing_action(
                                        current_df,
                                        action_to_execute,
                                        recommended_method
                                    )
                                    
                                    if exec_log['success']:
                                        current_df = cleaned_df  # Update for next action
                                        execution_results.append({
                                            'approval': approval,
                                            'action': action_to_execute,
                                            'method': recommended_method,
                                            'exec_log': exec_log,
                                            'success': True
                                        })
                                    else:
                                        execution_results.append({
                                            'approval': approval,
                                            'action': action_to_execute,
                                            'method': recommended_method,
                                            'exec_log': exec_log,
                                            'success': False
                                        })
                                except Exception as e:
                                    execution_results.append({
                                        'approval': approval,
                                        'action': action_to_execute,
                                        'method': recommended_method,
                                        'exec_log': {'success': False, 'message': str(e)},
                                        'success': False,
                                        'error': str(e)
                                    })
                                
                                # Update progress
                                progress_bar.progress((idx + 1) / len(selected_indices))
                        
                        # Step 2: Calculate quality score ONCE after all actions
                        status_text.text("📊 Calculating final quality score...")
                        final_quality_score = healing_agent.recalculate_quality_score(current_df)
                        
                        # Step 3: Batch write to database (single transaction is faster)
                        status_text.text("💾 Saving results to database...")
                        
                        approved_count = sum(1 for r in execution_results if r['success'])
                        failed_count = len(execution_results) - approved_count
                        
                        # Prepare all actions for batch logging (single DB transaction = MUCH faster)
                        batch_actions = []
                        for result_item in execution_results:
                            if result_item['success']:
                                # Log approval
                                batch_actions.append({
                                    'dataset_id': st.session_state.current_dataset_id,
                                    'agent_name': "HumanOperator",
                                    'action_type': f"APPROVED_{result_item['approval']['action_type']}",
                                    'action_details': f"Bulk approved {result_item['approval']['action_type']} - {result_item['approval']['impact']}",
                                    'execution_status': "COMPLETED",
                                    'human_approval': "APPROVED"
                                })
                                
                                # Log execution
                                batch_actions.append({
                                    'dataset_id': st.session_state.current_dataset_id,
                                    'agent_name': "DataHealingAgent",
                                    'action_type': f"EXECUTED_{result_item['action']['type']}",
                                    'action_details': f"Bulk executed {result_item['method']} - Records affected: {result_item['exec_log'].get('records_affected', 0)}",
                                    'execution_status': "COMPLETED",
                                    'human_approval': None
                                })
                            else:
                                # Log failure
                                batch_actions.append({
                                    'dataset_id': st.session_state.current_dataset_id,
                                    'agent_name': "DataHealingAgent",
                                    'action_type': f"FAILED_{result_item['action']['type']}",
                                    'action_details': f"Bulk execution failed: {result_item['exec_log'].get('message', 'Unknown error')}",
                                    'execution_status': "FAILED",
                                    'human_approval': None
                                })
                        
                        # Batch log all actions in a single transaction (10-20x faster than individual logs)
                        if batch_actions:
                            db.batch_log_agent_actions(batch_actions)
                        
                        # Store final cleaned data ONCE
                        if approved_count > 0:
                            db.update_dataset_quality_score(
                                st.session_state.current_dataset_id,
                                final_quality_score,
                                current_df,
                                "HEALING_IN_PROGRESS"
                            )
                            
                            # Update session state with final df
                            st.session_state.current_df = current_df
                            st.session_state.execution_logs = st.session_state.get('execution_logs', [])
                            st.session_state.execution_logs.extend([r['exec_log'] for r in execution_results if r['success']])
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show final results
                        if approved_count > 0:
                            st.success(f"✅ Bulk approval completed! {approved_count} actions processed successfully. Final quality score: {final_quality_score:.1f}/100")
                            if failed_count > 0:
                                st.warning(f"⚠️ {failed_count} actions failed during execution.")
                        else:
                            st.error(f"❌ All {failed_count} selected actions failed to execute.")
                        
                        # Clear selections and rerun
                        if 'selected_approvals' in st.session_state:
                            del st.session_state.selected_approvals
                        
                        import time
                        time.sleep(1)  # Brief pause to show completion message
                        st.rerun()

            with col3:
                if st.button("❌ Reject Selected", disabled=not st.session_state.get('selected_approvals', [])):
                    selected_indices = st.session_state.get('selected_approvals', [])
                    if selected_indices:
                        # Store bulk rejection info for feedback collection
                        st.session_state.bulk_rejection_indices = selected_indices
                        st.session_state.bulk_rejection_data = []
                        
                        for i in selected_indices:
                            if i < len(result['human_approvals_needed']):
                                approval = result['human_approvals_needed'][i]
                                st.session_state.bulk_rejection_data.append({
                                    'index': i,
                                    'action_type': approval['action_type'],
                                    'impact': approval['impact'],
                                    'agent_name': 'DataHealingAgent'
                                })
                        st.rerun()

            # Initialize selected approvals if not exists
            if 'selected_approvals' not in st.session_state:
                st.session_state.selected_approvals = []

            # Display individual actions with checkboxes
            for i, approval in enumerate(result['human_approvals_needed']):
                with st.expander(f"Action {i+1}: {approval['action_type'].replace('_', ' ').title()} ({approval.get('severity', 'MEDIUM')} Priority)", expanded=False):
                    # Checkbox for selection
                    is_selected = st.checkbox(
                        f"Select for bulk action",
                        key=f"select_approval_{i}",
                        value=i in st.session_state.selected_approvals
                    )

                    # Update selected list
                    if is_selected and i not in st.session_state.selected_approvals:
                        st.session_state.selected_approvals.append(i)
                    elif not is_selected and i in st.session_state.selected_approvals:
                        st.session_state.selected_approvals.remove(i)

                    # Action details
                    st.write(f"**Impact:** {approval['impact']}")
                    if approval.get('recommendation'):
                        st.write(f"**AI Recommendation:** {approval['recommendation']['description']}")

                    # Individual action buttons (still available)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Approve Only This", key=f"approve_single_{i}"):
                            # Execute single action (same logic as before)
                            approval_action_id = db.log_agent_action(
                                st.session_state.current_dataset_id,
                                "HumanOperator",
                                f"APPROVED_{approval['action_type']}",
                                f"Single approved {approval['action_type']} - {approval['impact']}",
                                "EXECUTING"
                            )

                            action_to_execute = result['healing_plan']['actions'][i]
                            recommended_method = approval['recommendation']['method'] if approval.get('recommendation') else 'default'

                            with st.spinner(f"Executing {approval['action_type']}..."):
                                # Execute healing action
                                cleaned_df, exec_log = healing_agent.execute_healing_action(
                                    st.session_state.current_df,
                                    action_to_execute,
                                    recommended_method
                                )

                                if exec_log['success']:
                                    # Update in-memory state
                                    st.session_state.current_df = cleaned_df
                                    st.session_state.execution_logs = st.session_state.get('execution_logs', [])
                                    st.session_state.execution_logs.append(exec_log)

                                    # Calculate quality score
                                    new_quality_score = healing_agent.recalculate_quality_score(cleaned_df)
                                    
                                    # Batch database operations (faster)
                                    db.update_agent_action(approval_action_id, "COMPLETED", "APPROVED")
                                    db.log_agent_action(
                                        st.session_state.current_dataset_id,
                                        "DataHealingAgent",
                                        f"EXECUTED_{action_to_execute['type']}",
                                        f"Single executed {recommended_method} - Records affected: {exec_log.get('records_affected', 0)}, New Quality Score: {new_quality_score:.1f}/100",
                                        "COMPLETED"
                                    )
                                    
                                    # Store DataFrame LAST (most expensive operation)
                                    db.update_dataset_quality_score(
                                        st.session_state.current_dataset_id,
                                        new_quality_score,
                                        cleaned_df,
                                        "HEALING_IN_PROGRESS"
                                    )

                                    st.success(f"✅ Action completed successfully! Quality score: {new_quality_score:.1f}/100")
                                    st.rerun()
                                else:
                                    db.update_agent_action(approval_action_id, "FAILED", "APPROVED")
                                    db.log_agent_action(
                                        st.session_state.current_dataset_id,
                                        "DataHealingAgent",
                                        f"FAILED_{action_to_execute['type']}",
                                        f"Single execution failed: {exec_log['message']}",
                                        "FAILED"
                                    )
                                    st.error(f"❌ Execution failed: {exec_log['message']}")

                    with col2:
                        # Enhanced rejection with alternative solution option
                        reject_col1, reject_col2 = st.columns([1, 2])
                        with reject_col1:
                            if st.button(f"❌ Reject Only This", key=f"reject_single_{i}"):
                                # Store rejection reason in session state for feedback
                                st.session_state[f"rejection_feedback_{i}"] = {
                                    'action_type': approval['action_type'],
                                    'impact': approval['impact'],
                                    'agent_name': 'DataHealingAgent',
                                    'action_id': None  # Will be set after logging
                                }
                                st.rerun()
                        
                        # Show alternative solution input if rejection was clicked
                        if f"rejection_feedback_{i}" in st.session_state:
                            with reject_col2:
                                st.markdown("#### 💡 Provide Alternative Solution")
                                alternative_method = st.selectbox(
                                    "Alternative Healing Method",
                                    options=[
                                        "None (just reject)",
                                        "FILL_MEAN", "FILL_MEDIAN", "FILL_MODE", "FILL_FORWARD", "FILL_BACKWARD",
                                        "DROP_ROWS", "DROP_COLUMNS", 
                                        "OUTLIER_IQR", "OUTLIER_ZSCORE", "OUTLIER_PERCENTILE",
                                        "DUPLICATE_REMOVE", "DUPLICATE_KEEP_FIRST", "DUPLICATE_KEEP_LAST",
                                        "CUSTOM_TRANSFORMATION"
                                    ],
                                    key=f"alt_method_{i}",
                                    help="Choose an alternative healing approach or select 'None' to just reject"
                                )
                                
                                custom_notes = st.text_area(
                                    "Additional Notes (Optional)",
                                    placeholder="Explain why this alternative is better, or provide custom instructions...",
                                    height=60,
                                    key=f"custom_notes_{i}"
                                )
                                
                                feedback_rating = st.slider(
                                    "How confident are you in this alternative?",
                                    min_value=1, max_value=5, value=4,
                                    help="1 = Not confident, 5 = Very confident",
                                    key=f"rating_{i}"
                                )
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button("✅ Submit Alternative & Reject", 
                                               key=f"submit_alt_{i}", type="primary"):
                                        # Log the rejection with alternative solution
                                        rejection_details = st.session_state[f"rejection_feedback_{i}"]
                                        
                                        action_id = db.log_agent_action(
                                            st.session_state.current_dataset_id,
                                            "HumanOperator",
                                            f"REJECTED_WITH_ALTERNATIVE_{rejection_details['action_type']}",
                                            f"Rejected {rejection_details['action_type']} - Impact: {rejection_details['impact']} - Alternative: {alternative_method}",
                                            "REJECTED"
                                        )
                                        
                                        # Store human feedback with alternative solution
                                        if orchestrator and hasattr(orchestrator, 'learning_system'):
                                            feedback_text = f"Rejected: {rejection_details['action_type']}. Alternative: {alternative_method}"
                                            if custom_notes.strip():
                                                feedback_text += f". Notes: {custom_notes}"
                                            
                                            orchestrator.learning_system.store_human_feedback(
                                                agent_name="DataHealingAgent",
                                                action_id=action_id,
                                                feedback_type="rejection_with_alternative",
                                                feedback_text=feedback_text,
                                                rating=feedback_rating,
                                                context=f"Dataset: {st.session_state.get('dataset_name', 'Unknown')} - Column: {approval.get('column', 'N/A')}"
                                            )
                                        
                                        st.success(f"✅ Rejection logged with alternative solution: {alternative_method}")
                                        
                                        # Clear the feedback state
                                        del st.session_state[f"rejection_feedback_{i}"]
                                        st.rerun()
                                        
                                with col_b:
                                    if st.button("❌ Just Reject (No Alternative)", 
                                               key=f"just_reject_{i}"):
                                        # Simple rejection without alternative
                                        rejection_details = st.session_state[f"rejection_feedback_{i}"]
                                        
                                        db.log_agent_action(
                                            st.session_state.current_dataset_id,
                                            "HumanOperator",
                                            f"REJECTED_{rejection_details['action_type']}",
                                            f"Single rejected {rejection_details['action_type']} - Impact: {rejection_details['impact']}",
                                            "REJECTED"
                                        )
                                        
                                        st.warning(f"❌ Action rejected. Issue logged for manual review.")
                                        
                                        # Clear the feedback state
                                        del st.session_state[f"rejection_feedback_{i}"]
                                        st.rerun()

            # Bulk Rejection Feedback Collection
            if 'bulk_rejection_indices' in st.session_state and st.session_state.bulk_rejection_indices:
                st.markdown("---")
                with st.expander("💡 Provide Alternative Solutions for Rejected Actions", expanded=True):
                    st.info(f"You rejected {len(st.session_state.bulk_rejection_indices)} actions. Please provide alternative solutions below:")
                    
                    # Collect feedback for each rejected action
                    bulk_feedback = {}
                    for i, rejection_data in enumerate(st.session_state.bulk_rejection_data):
                        st.markdown(f"**Action {i+1}: {rejection_data['action_type'].replace('_', ' ').title()} ({rejection_data.get('severity', 'MEDIUM')} Priority)**")
                        st.write(f"Impact: {rejection_data['impact']}")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            alt_method = st.selectbox(
                                f"Alternative Method",
                                options=[
                                    "None (just reject)",
                                    "FILL_MEAN", "FILL_MEDIAN", "FILL_MODE", "FILL_FORWARD", "FILL_BACKWARD",
                                    "DROP_ROWS", "DROP_COLUMNS", 
                                    "OUTLIER_IQR", "OUTLIER_ZSCORE", "OUTLIER_PERCENTILE",
                                    "DUPLICATE_REMOVE", "DUPLICATE_KEEP_FIRST", "DUPLICATE_KEEP_LAST",
                                    "CUSTOM_TRANSFORMATION"
                                ],
                                key=f"bulk_alt_method_{i}",
                                help="Choose an alternative healing approach"
                            )
                        
                        with col2:
                            notes = st.text_area(
                                f"Notes (Optional)",
                                placeholder="Explain why this alternative is better...",
                                height=60,
                                key=f"bulk_notes_{i}"
                            )
                        
                        rating = st.slider(
                            f"Confidence in Alternative",
                            min_value=1, max_value=5, value=4,
                            key=f"bulk_rating_{i}"
                        )
                        
                        bulk_feedback[i] = {
                            'alternative_method': alt_method,
                            'notes': notes,
                            'rating': rating,
                            'rejection_data': rejection_data
                        }
                    
                    # Submit all bulk feedback
                    if st.button("✅ Submit All Alternatives & Complete Rejection", type="primary"):
                        submitted_count = 0
                        for i, feedback in bulk_feedback.items():
                            rejection_data = feedback['rejection_data']
                            
                            # Log the rejection with alternative
                            action_id = db.log_agent_action(
                                st.session_state.current_dataset_id,
                                "HumanOperator",
                                f"BULK_REJECTED_WITH_ALTERNATIVE_{rejection_data['action_type']}",
                                f"Bulk rejected {rejection_data['action_type']} - Impact: {rejection_data['impact']} - Alternative: {feedback['alternative_method']}",
                                "REJECTED"
                            )
                            
                            # Store human feedback
                            if orchestrator and hasattr(orchestrator, 'learning_system'):
                                feedback_text = f"Bulk rejected: {rejection_data['action_type']}. Alternative: {feedback['alternative_method']}"
                                if feedback['notes'].strip():
                                    feedback_text += f". Notes: {feedback['notes']}"
                                
                                orchestrator.learning_system.store_human_feedback(
                                    agent_name="DataHealingAgent",
                                    action_id=action_id,
                                    feedback_type="bulk_rejection_with_alternative",
                                    feedback_text=feedback_text,
                                    rating=feedback['rating'],
                                    context=f"Bulk rejection - Dataset: {st.session_state.get('dataset_name', 'Unknown')}"
                                )
                            
                            submitted_count += 1
                        
                        st.success(f"✅ All {submitted_count} rejections completed with alternative solutions provided!")
                        
                        # Clear bulk rejection state
                        del st.session_state.bulk_rejection_indices
                        del st.session_state.bulk_rejection_data
                        if 'selected_approvals' in st.session_state:
                            del st.session_state.selected_approvals
                        st.rerun()
                    
                    # Option to just reject without alternatives
                    if st.button("❌ Just Reject All (No Alternatives)"):
                        rejected_count = 0
                        for rejection_data in st.session_state.bulk_rejection_data:
                            db.log_agent_action(
                                st.session_state.current_dataset_id,
                                "HumanOperator",
                                f"BULK_REJECTED_{rejection_data['action_type']}",
                                f"Bulk rejected {rejection_data['action_type']} - Impact: {rejection_data['impact']}",
                                "REJECTED"
                            )
                            rejected_count += 1
                        
                        st.warning(f"❌ {rejected_count} actions rejected and logged for manual review.")
                        
                        # Clear bulk rejection state
                        del st.session_state.bulk_rejection_indices
                        del st.session_state.bulk_rejection_data
                        if 'selected_approvals' in st.session_state:
                            del st.session_state.selected_approvals
                        st.rerun()

        # Execution Logs
        if 'execution_logs' in st.session_state:
            st.markdown("### 📝 Healing Action History")
            
            # Check if all healing actions are complete
            all_actions_complete = len(st.session_state.execution_logs) >= len(result.get('human_approvals_needed', []))
            
            if all_actions_complete:
                # Final quality score calculation
                with st.spinner("📊 Calculating final quality score..."):
                    final_quality_score = healing_agent.recalculate_quality_score(st.session_state.current_df)
                
                # Update the processing_result with the final quality report
                if 'processing_result' in st.session_state:
                    # Create updated quality report with final score
                    final_quality_report = st.session_state.processing_result.get('quality_report', {}).copy()
                    final_quality_report['quality_score'] = final_quality_score
                    final_quality_report['healing_completed'] = True
                    final_quality_report['final_quality_score'] = final_quality_score
                    
                    # Update the processing_result
                    st.session_state.processing_result['quality_report'] = final_quality_report
                    st.session_state.processing_result['final_quality_score'] = final_quality_score
                
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

    # Monitoring Dashboard - Collapsible
    st.markdown("---")
    with st.expander("📈 Pipeline Monitoring", expanded=False):
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

    # Real-Time Dashboard - Collapsible
    with st.expander("⚡ Real-Time Activity", expanded=False):
        # System Status
        monitor_status = rt_monitor.get_status()
        
        if monitor_status['running']:
            st.success(f"🟢 **Real-Time Monitor Active** - Running for {str(monitor_status['uptime']).split('.')[0] if monitor_status['uptime'] else '0:00:00'}")
        else:
            st.error("🔴 **Real-Time Monitor Stopped** - Start it from the sidebar to see live data")
        
        # Real-time metrics
        metrics = rt_monitor.get_system_metrics()
        api_stats = rt_monitor.get_api_fetch_stats()
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        with col6:
            st.metric("API Fetches (1h)", api_stats['recent_fetches'])
        
        # Live Activity Feed
        st.subheader("📡 Live Activity Feed")
        
        recent_actions = rt_monitor.get_recent_activity()
        
        if recent_actions:
            for action in recent_actions[-10:]:  # Show last 10 activities
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{action['timestamp'].strftime('%H:%M:%S')}**: {action['message']}")
                    with col2:
                        # Show relative time
                        time_diff = datetime.now() - action['timestamp']
                        if time_diff.seconds < 60:
                            st.write(f"{time_diff.seconds}s ago")
                        elif time_diff.seconds < 3600:
                            st.write(f"{time_diff.seconds//60}m ago")
                        else:
                            st.write(f"{time_diff.seconds//3600}h ago")
                    
                    st.divider()
        else:
            st.info("No recent activity to display. The monitor will show activity here as it processes data.")

    # Action Log - Collapsible
    with st.expander("📝 Action Log", expanded=False):
        with st.spinner("📊 Loading action history..."):
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
            
            # Display recent actions
            st.subheader("Recent Actions")
            
            # Show last 10 actions
            for _, action in agent_actions.head(10).iterrows():
                # Color coding based on status
                status_color = {
                    'COMPLETED': '🟢',
                    'EXECUTING': '🟡', 
                    'PENDING_APPROVAL': '🟡',
                    'FAILED': '🔴',
                    'REJECTED': '🔴',
                    'AUTO_APPROVED': '🟢'
                }.get(action['ExecutionStatus'], '⚫')
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 2])
                    with col1:
                        st.write(f"{status_color} **{action['AgentName']}**: {action['ActionType']}")
                    with col2:
                        st.write(f"**{action['ExecutionStatus']}**")
                    with col3:
                        st.write(f"**{action['ActionTimestamp'].strftime('%H:%M:%S')}**")
                    
                    # Show details in a smaller text
                    details = str(action['ActionDetails'])
                    if len(details) > 100:
                        details = details[:100] + "..."
                    st.caption(f"Details: {details}")
                    
                    st.divider()
        else:
            st.info("No agent actions logged yet.")

    # Download Healed Data Section
    if 'processing_result' in st.session_state and 'current_df' in st.session_state:
        st.markdown("---")
        with st.expander("💾 Download Healed Data", expanded=False):
            st.markdown("**Download your processed and healed dataset:**")
            
            healed_df = st.session_state.current_df
            dataset_name = st.session_state.get('dataset_name', 'processed_data')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Download
                with st.spinner("📄 Preparing CSV file..."):
                    csv_data = healed_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"{dataset_name}_healed.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            
            with col2:
                # Excel Download
                from io import BytesIO
                with st.spinner("📊 Preparing Excel file..."):
                    buffer = BytesIO()
                    healed_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                st.download_button(
                    label="📊 Download as Excel",
                    data=buffer,
                    file_name=f"{dataset_name}_healed.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            
            with col3:
                # JSON Download
                with st.spinner("🔧 Preparing JSON file..."):
                    json_data = healed_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="🔧 Download as JSON",
                    data=json_data,
                    file_name=f"{dataset_name}_healed.json",
                    mime="application/json",
                    key="download_json"
                )
            
            # Database Save Option
            st.markdown("### 💾 Database Operations")
            col_db1, col_db2 = st.columns(2)
            
            with col_db1:
                if st.button("💾 Save to Database", key="save_to_db"):
                    try:
                        # Save healed data to database
                        db_dataset_id = db.store_dataset(
                            healed_df,
                            f"HEALED_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "HEALED",
                            {
                                "original_file": st.session_state.get('file_dataset_name', dataset_name),
                                "healing_performed": True,
                                "quality_report": st.session_state.processing_result.get('quality_report', {}),
                                "validation_report": st.session_state.processing_result.get('validation_report', {})
                            }
                        )
                        st.success(f"✅ Healed data saved to database! Dataset ID: {db_dataset_id}")
                    except Exception as e:
                        st.error(f"❌ Failed to save to database: {str(e)}")
            
            with col_db2:
                if st.button("📊 View in Database Browser", key="view_in_db"):
                    st.info("💡 Database browser feature coming soon! For now, use the 'Database Browser' page to explore saved datasets.")
            
            # Dataset Summary
            st.markdown("### 📊 Dataset Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(healed_df))
            with col2:
                st.metric("Total Columns", len(healed_df.columns))
            with col3:
                # Use final quality score if healing is completed, otherwise use original
                quality_report = st.session_state.processing_result.get('quality_report', {})
                if quality_report.get('healing_completed', False):
                    quality_score = quality_report.get('final_quality_score', quality_report.get('quality_score', 0))
                else:
                    quality_score = quality_report.get('quality_score', 0)
                st.metric("Quality Score", f"{quality_score:.1f}%")
            with col4:
                issues_count = len(st.session_state.processing_result.get('quality_report', {}).get('issues', []))
                st.metric("Issues Resolved", issues_count)

    # Real-Time API Data Feed Section
    st.markdown("---")
    with st.expander("🔄 Real-Time API Data Feed", expanded=False):
        st.markdown("**Connect to live APIs for continuous data ingestion:**")
        st.markdown("💡 **Automatic fetching is active** - APIs will be fetched based on their configured intervals when the real-time monitor is running.")
        
        # Show monitor status
        monitor_status = rt_monitor.get_status()
        if monitor_status['running']:
            st.success("🟢 Real-time monitor is running - automatic API fetching enabled")
        else:
            st.warning("🟡 Real-time monitor is stopped - APIs will only be fetched manually")
        
        # API Connection Setup
        col1, col2 = st.columns([2, 1])
        with col1:
            # Initialize session state for demo APIs
            if 'demo_api_url' not in st.session_state:
                st.session_state.demo_api_url = ""
            if 'demo_api_name' not in st.session_state:
                st.session_state.demo_api_name = ""
            
            api_url = st.text_input(
                "API Endpoint URL",
                value=st.session_state.demo_api_url,
                placeholder="https://api.example.com/data",
                help="Enter the API endpoint URL for data fetching"
            )
            api_name = st.text_input(
                "API Source Name",
                value=st.session_state.demo_api_name,
                placeholder="e.g., Sales API, Weather API",
                help="Give this API source a descriptive name"
            )
            
            # Demo API suggestions
            st.markdown("**💡 Demo APIs to try:**")
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                if st.button("🌤️ Weather API (Open-Meteo)"):
                    st.session_state.demo_api_url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&hourly=temperature_2m,relative_humidity_2m,precipitation"
                    st.session_state.demo_api_name = "NYC Weather API"
                    st.rerun()
            with demo_col2:
                if st.button("📊 JSON Placeholder"):
                    st.session_state.demo_api_url = "https://jsonplaceholder.typicode.com/posts"
                    st.session_state.demo_api_name = "Demo Posts API"
                    st.rerun()
        
        with col2:
            st.markdown("**Fetch Interval**")
            interval_col1, interval_col2 = st.columns([1, 1])
            
            with interval_col1:
                interval_value = st.number_input(
                    "Value",
                    min_value=1,
                    max_value=999,
                    value=30,
                    step=1,
                    help="Enter the interval value"
                )
            
            with interval_col2:
                interval_unit = st.selectbox(
                    "Unit",
                    options=["seconds", "minutes", "hours"],
                    index=0,
                    help="Select time unit"
                )
            
            # Convert to seconds
            if interval_unit == "minutes":
                fetch_interval = interval_value * 60
            elif interval_unit == "hours":
                fetch_interval = interval_value * 3600
            else:
                fetch_interval = interval_value
            
            st.caption(f"⏱️ Fetch every {fetch_interval} seconds ({interval_value} {interval_unit})")
            st.write("")  # Spacing
            if st.button("🔗 Connect API Source", type="primary"):
                if api_url and api_name:
                    try:
                        # Test the API connection
                        import requests
                        response = requests.get(api_url, timeout=10)
                        
                        if response.status_code == 200:
                            # Register the API source
                            source_id = db.register_api_source(api_name, api_url, fetch_interval)
                            
                            # Try to fetch initial data
                            try:
                                data = response.json()
                                
                                # Handle different response structures
                                if isinstance(data, dict):
                                    # If response is a dict with a data key, use that
                                    if 'data' in data and isinstance(data['data'], list):
                                        data = data['data']
                                    elif 'results' in data and isinstance(data['results'], list):
                                        data = data['results']
                                    elif 'records' in data and isinstance(data['records'], list):
                                        data = data['records']
                                    else:
                                        # Single record dict, convert to list
                                        data = [data]
                                
                                if isinstance(data, list) and len(data) > 0:
                                    # Store the data
                                    db.store_api_snapshot(source_id, data, "INITIAL_FETCH")
                                    st.success(f"✅ API '{api_name}' connected successfully! Fetched {len(data)} records.")
                                    
                                    # Show data preview
                                    st.markdown("#### 📊 Data Preview")
                                    preview_df = pd.DataFrame(data[:5])  # Show first 5 records
                                    st.dataframe(preview_df, use_container_width=True)
                                    st.caption(f"Showing 5 of {len(data)} records")
                                else:
                                    st.warning(f"⚠️ API connected but returned empty data.")
                            except Exception as e:
                                st.warning(f"⚠️ API connected but data parsing failed: {str(e)}")
                        else:
                            st.error(f"❌ API connection failed: HTTP {response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ API connection failed: {str(e)}")
                else:
                    st.error("❌ Please provide both API URL and name.")
        
        # Connected APIs List
        st.markdown("### 🔗 Connected API Sources")
        
        # Bulk actions
        try:
            api_sources = db.list_api_sources()
            if not api_sources.empty:
                col_bulk, col_spacer = st.columns([1, 3])
                with col_bulk:
                    # Remove all API sources button with confirmation
                    remove_all_key = "remove_all_confirm"
                    if remove_all_key not in st.session_state:
                        st.session_state[remove_all_key] = False
                    
                    if not st.session_state[remove_all_key]:
                        if st.button("🗑️ Remove All Sources", help="Remove all connected API sources and their data"):
                            st.session_state[remove_all_key] = True
                            st.rerun()
                    else:
                        st.error("⚠️ Really remove ALL API sources? This will delete all fetch history and cannot be undone!")
                        col_confirm_all, col_cancel_all = st.columns(2)
                        with col_confirm_all:
                            if st.button("✅ Yes, Remove All", key="confirm_remove_all", type="primary"):
                                try:
                                    removed_count = 0
                                    for _, source in api_sources.iterrows():
                                        if db.remove_api_source(source['ID']):
                                            removed_count += 1
                                            # Clear any cached data for this source
                                            if f"api_preview_{source['ID']}" in st.session_state:
                                                del st.session_state[f"api_preview_{source['ID']}"]
                                    
                                    if removed_count > 0:
                                        st.success(f"✅ Removed {removed_count} API sources and all their data!")
                                        st.session_state[remove_all_key] = False
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to remove API sources")
                                except Exception as e:
                                    st.error(f"❌ Error removing API sources: {str(e)}")
                        
                        with col_cancel_all:
                            if st.button("❌ Cancel", key="cancel_remove_all"):
                                st.session_state[remove_all_key] = False
                                st.rerun()
                
                st.markdown("---")
            
            if not api_sources.empty:
                for _, source in api_sources.iterrows():
                    # Use container instead of nested expander
                    with st.container():
                        st.markdown(f"**🔗 {source['SourceName']}**")
                        st.caption(f"URL: {source['ApiUrl']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            status_icon = "🟢" if source['IsActive'] else "🔴"
                            st.write(f"**Status:** {status_icon} {'Active' if source['IsActive'] else 'Inactive'}")
                        with col2:
                            interval_min = source['FetchInterval'] // 60
                            auto_fetch = "Auto-fetch enabled" if source['FetchInterval'] > 0 else "Manual fetch only"
                            st.write(f"**Interval:** {interval_min} min ({auto_fetch})")
                        with col3:
                            last_fetch = source['LastFetchTimestamp']
                            if pd.notna(last_fetch):
                                time_since = (datetime.now() - last_fetch).total_seconds()
                                if time_since < 60:
                                    time_str = f"{int(time_since)}s ago"
                                elif time_since < 3600:
                                    time_str = f"{int(time_since//60)}m ago"
                                else:
                                    time_str = last_fetch.strftime('%H:%M:%S')
                                st.write(f"**Last Fetch:** {time_str}")
                            else:
                                st.write("**Last Fetch:** Never")
                        
                        # Show next fetch time if active
                        if source['IsActive'] and source['FetchInterval'] > 0 and pd.notna(last_fetch):
                            next_fetch = last_fetch + timedelta(seconds=source['FetchInterval'])
                            time_to_next = (next_fetch - datetime.now()).total_seconds()
                            if time_to_next > 0:
                                if time_to_next < 60:
                                    next_str = f"{int(time_to_next)}s"
                                elif time_to_next < 3600:
                                    next_str = f"{int(time_to_next//60)}m"
                                else:
                                    next_str = next_fetch.strftime('%H:%M:%S')
                                st.info(f"⏰ Next auto-fetch in: {next_str}")
                        
                        # Action buttons
                        col_fetch, col_remove = st.columns([1, 1])
                        
                        with col_fetch:
                            # Manual fetch button
                            if st.button(f"🔄 Fetch Now", key=f"fetch_{source['ID']}"):
                                try:
                                    # Use the improved fetch_and_store_api method that handles nested JSON
                                    result = db.fetch_and_store_api(
                                        source_id=source['ID'],
                                        url=source['ApiUrl'],
                                        method='GET',
                                        data_format='json',
                                        timeout=20
                                    )
                                    
                                    # Result can be a DataFrame or raw JSON
                                    if isinstance(result, pd.DataFrame):
                                        st.session_state[f"api_preview_{source['ID']}"] = result
                                        st.success(f"✅ Data fetched successfully! ({len(result)} records)")
                                        st.rerun()
                                    elif isinstance(result, dict) or isinstance(result, list):
                                        # Try to convert to DataFrame
                                        try:
                                            df = pd.DataFrame(result) if isinstance(result, list) else pd.DataFrame([result])
                                            st.session_state[f"api_preview_{source['ID']}"] = df
                                            st.success(f"✅ Data fetched successfully! ({len(df)} records)")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Could not convert API response to table: {str(e)}")
                                    else:
                                        st.error("❌ API returned unexpected data format")
                                except Exception as e:
                                    st.error(f"❌ Fetch failed: {str(e)}")
                        
                        with col_remove:
                            # Remove API source button with confirmation
                            remove_key = f"remove_confirm_{source['ID']}"
                            if remove_key not in st.session_state:
                                st.session_state[remove_key] = False
                            
                            if not st.session_state[remove_key]:
                                if st.button(f"🗑️ Remove", key=f"remove_{source['ID']}", 
                                           help=f"Remove {source['SourceName']} API source"):
                                    st.session_state[remove_key] = True
                                    st.rerun()
                            else:
                                st.warning(f"⚠️ Really remove '{source['SourceName']}'? This will delete all fetch history!")
                                col_confirm, col_cancel = st.columns(2)
                                with col_confirm:
                                    if st.button("✅ Yes, Remove", key=f"confirm_remove_{source['ID']}", type="primary"):
                                        try:
                                            if db.remove_api_source(source['ID']):
                                                st.success(f"✅ Removed API source '{source['SourceName']}' and all its data!")
                                                # Clear any cached data for this source
                                                if f"api_preview_{source['ID']}" in st.session_state:
                                                    del st.session_state[f"api_preview_{source['ID']}"]
                                                st.session_state[remove_key] = False
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to remove API source")
                                        except Exception as e:
                                            st.error(f"❌ Error removing API source: {str(e)}")
                                
                                with col_cancel:
                                    if st.button("❌ Cancel", key=f"cancel_remove_{source['ID']}"):
                                        st.session_state[remove_key] = False
                                        st.rerun()
                        
                        # Add separator between API sources
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"Error loading API sources: {str(e)}")
        
        # API Data Healing Section
        st.markdown("### 🩺 Heal Fetched API Data")
        
        # Check for any API preview data that can be healed
        api_preview_keys = [key for key in st.session_state.keys() if key.startswith('api_preview_')]
        
        if api_preview_keys:
            st.markdown("**Available API datasets for healing:**")
            
            for preview_key in api_preview_keys:
                source_id = preview_key.replace('api_preview_', '')
                api_df = st.session_state[preview_key]
                
                # Get source name
                try:
                    sources_df = db.list_api_sources()
                    source_row = sources_df[sources_df['ID'] == int(source_id)]
                    if not source_row.empty:
                        source_name = source_row.iloc[0]['SourceName']
                    else:
                        source_name = f"API Source {source_id}"
                except:
                    source_name = f"API Source {source_id}"
                
                with st.expander(f"🩺 Heal: {source_name} ({len(api_df)} records)", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records", len(api_df))
                    with col2:
                        st.metric("Columns", len(api_df.columns))
                    with col3:
                        st.metric("Missing Values", api_df.isnull().sum().sum())
                    
                    # Show preview
                    st.dataframe(api_df.head(5), use_container_width=True)
                    
                    # Quality Analysis Section
                    st.markdown("### 📊 Data Quality Assessment")
                    
                    # Analyze quality for display
                    try:
                        quality_report = quality_agent.analyze_dataset(api_df, f"API_{source_name}")
                        quality_score = quality_report.get('quality_score', 0)
                        
                        # Quality score display
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            if quality_score >= 90:
                                st.metric("Quality Score", f"{quality_score:.1f}%", "Excellent")
                                st.success("🟢 High Quality Data")
                            elif quality_score >= 70:
                                st.metric("Quality Score", f"{quality_score:.1f}%", "Good")
                                st.info("🟡 Moderate Quality")
                            else:
                                st.metric("Quality Score", f"{quality_score:.1f}%", "Needs Healing")
                                st.warning("🔴 Low Quality Data")
                        
                        with col_q2:
                            issues_count = len(quality_report.get('issues', []))
                            st.metric("Issues Found", issues_count)
                        
                        with col_q3:
                            if issues_count > 0:
                                st.metric("Status", "Needs Attention")
                            else:
                                st.metric("Status", "Ready to Use")
                        
                        # Show top issues if any
                        if issues_count > 0:
                            st.markdown("**Top Issues Detected:**")
                            issues_list = quality_report.get('issues', [])[:3]  # Show top 3
                            for issue in issues_list:
                                issue_type = issue.get('type', 'Unknown')
                                severity = issue.get('severity', 'medium')
                                description = issue.get('description', 'Data quality issue detected')
                                
                                if severity == 'high':
                                    st.error(f"🔴 {issue_type}: {description}")
                                elif severity == 'medium':
                                    st.warning(f"🟡 {issue_type}: {description}")
                                else:
                                    st.info(f"ℹ️ {issue_type}: {description}")
                    
                    except Exception as e:
                        st.error(f"❌ Quality analysis failed: {str(e)}")
                        quality_score = 0
                    
                    # Healing button
                    heal_key = f"heal_api_{source_id}"
                    
                    # Determine button text and type based on quality
                    if 'quality_score' in locals() and quality_score >= 90:
                        button_text = f"✨ Optimize This High-Quality API Data"
                        button_type = "secondary"
                    elif 'quality_score' in locals() and quality_score >= 70:
                        button_text = f"🩺 Improve This API Data Quality"
                        button_type = "primary"
                    else:
                        button_text = f"🩺 Auto-Heal This API Data (Quality: {quality_score:.1f}%)"
                        button_type = "primary"
                    
                    if st.button(button_text, key=heal_key, type=button_type):
                        # Create progress tracking
                        progress_container = st.container()
                        status_container = st.container()
                        
                        with progress_container:
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                        
                        with status_container:
                            status_text = st.empty()
                        
                        try:
                            # Step 1: Dataset Storage (10%)
                            progress_text.text("📦 Storing API dataset...")
                            progress_bar.progress(10)
                            status_text.info("🔄 Step 1/6: Dataset Storage - Saving API data securely...")
                            
                            # Clean the DataFrame to ensure it can be serialized to JSON
                            # Remove columns that contain complex objects (dicts, lists)
                            cleaned_df = api_df.copy()
                            for col in cleaned_df.columns:
                                if cleaned_df[col].dtype == 'object':
                                    # Check if any values in this column are dicts or lists
                                    sample_values = cleaned_df[col].dropna().head(3)
                                    if any(isinstance(val, (dict, list)) for val in sample_values):
                                        # Convert complex objects to strings
                                        cleaned_df[col] = cleaned_df[col].astype(str)
                            
                            # Store in database with API source context
                            dataset_id = db.store_dataset(
                                f"API_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                cleaned_df
                            )
                            
                            # Step 2: Quality Analysis (30%)
                            progress_text.text("🔍 Analyzing API data quality...")
                            progress_bar.progress(30)
                            status_text.info("🔄 Step 2/6: Quality Analysis - Scanning for issues...")
                            
                            quality_report = quality_agent.analyze_dataset(api_df, f"API_{source_name}")
                            
                            # Step 3: Healing Plan Generation (50%)
                            progress_text.text("🧠 Generating healing plan...")
                            progress_bar.progress(50)
                            status_text.info("🔄 Step 3/6: AI Planning - Creating healing strategies...")
                            
                            healing_plan = healing_agent.propose_healing_actions(api_df, quality_report)
                            
                            # Step 4: Auto-Execution (70%)
                            progress_text.text("🔧 Auto-executing healing actions...")
                            progress_bar.progress(70)
                            status_text.info("🔄 Step 4/6: Auto-Healing - Applying fixes automatically...")
                            
                            # Choose processing method based on dataset size
                            if len(api_df) > 100000:  # Use chunked processing for large datasets
                                status_text.info(f"🔄 Step 4/6: Large Dataset Processing - Using chunked healing for {len(api_df)} rows...")
                                healed_df, healing_summary = healing_agent.execute_healing_actions_chunked(
                                    api_df, healing_plan, chunk_size=50000
                                )
                                st.info(f"📊 Processed in {healing_summary.get('total_chunks', 1)} chunks, {healing_summary.get('total_records_affected', 0)} records affected")
                            else:
                                # Execute all healing actions from the plan
                                healed_df = api_df.copy()
                                for action in healing_plan.get('actions', []):
                                    method = action.get('recommended_method', action.get('method', 'DELETE_ROWS'))
                                    healed_df, exec_log = healing_agent.execute_healing_action(healed_df, action, method)
                            
                            # Step 5: Validation (90%)
                            progress_text.text("✅ Validating healed data...")
                            progress_bar.progress(90)
                            status_text.info("🔄 Step 5/6: Validation - Ensuring data quality...")
                            
                            # Create validation report by comparing before/after quality
                            final_quality_report = quality_agent.analyze_dataset(healed_df, f"HEALED_API_{source_name}")
                            
                            validation_report = {
                                'original_quality_score': quality_report.get('quality_score', 0),
                                'final_quality_score': final_quality_report.get('quality_score', 0),
                                'quality_improvement': final_quality_report.get('quality_score', 0) - quality_report.get('quality_score', 0),
                                'original_issues_count': len(quality_report.get('issues', [])),
                                'final_issues_count': len(final_quality_report.get('issues', [])),
                                'issues_resolved': len(quality_report.get('issues', [])) - len(final_quality_report.get('issues', [])),
                                'validation_timestamp': datetime.now().isoformat(),
                                'healing_actions_executed': len(healing_plan.get('actions', [])),
                                'validation_status': 'PASSED' if final_quality_report.get('quality_score', 0) >= quality_report.get('quality_score', 0) else 'IMPROVEMENT_NEEDED'
                            }
                            
                            # Step 6: Final Storage (100%)
                            progress_text.text("💾 Finalizing results...")
                            progress_bar.progress(100)
                            status_text.success("✅ Step 6/6: Complete - API data healed successfully!")
                            
                            # Store healed result
                            st.session_state[f'api_healed_{source_id}'] = healed_df
                            st.session_state[f'api_validation_{source_id}'] = validation_report
                            st.session_state[f'api_quality_{source_id}'] = quality_report
                            
                            st.success(f"🎉 API data '{source_name}' healed successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Healing failed: {str(e)}")
                            progress_bar.progress(100)
                            progress_text.text("❌ Healing failed")
                    
                    # Show healed results if available
                    healed_key = f'api_healed_{source_id}'
                    if healed_key in st.session_state:
                        st.success("✅ **Data Healed Successfully!**")
                        
                        healed_df = st.session_state[healed_key]
                        validation_report = st.session_state.get(f'api_validation_{source_id}', {})
                        
                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📊 Before Healing:**")
                            st.metric("Missing Values", api_df.isnull().sum().sum())
                            st.metric("Duplicates", api_df.duplicated().sum())
                        
                        with col2:
                            st.markdown("**✨ After Healing:**")
                            st.metric("Missing Values", healed_df.isnull().sum().sum())
                            st.metric("Duplicates", healed_df.duplicated().sum())
                        
                        # Show healed data preview
                        st.markdown("**🔎 Healed Data Preview:**")
                        st.dataframe(healed_df.head(10), use_container_width=True)
                        
                        # Download and Database options
                        st.markdown("### 💾 Export Healed API Data")
                        col_dl1, col_dl2, col_dl3 = st.columns(3)
                        
                        with col_dl1:
                            # CSV Download
                            csv_data = healed_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv_data,
                                file_name=f"{source_name}_healed.csv",
                                mime="text/csv",
                                key=f"api_download_csv_{source_id}"
                            )
                        
                        with col_dl2:
                            # Excel Download
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                healed_df.to_excel(writer, sheet_name='Healed_Data', index=False)
                                # Add validation sheet
                                if validation_report:
                                    pd.DataFrame([validation_report]).to_excel(writer, sheet_name='Validation_Report', index=False)
                            excel_data = excel_buffer.getvalue()
                            
                            st.download_button(
                                label="📊 Download Excel",
                                data=excel_data,
                                file_name=f"{source_name}_healed.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"api_download_excel_{source_id}"
                            )
                        
                        with col_dl3:
                            # Send to Database
                            if st.button("💾 Send to Database", key=f"api_to_db_{source_id}"):
                                try:
                                    # Store healed data in database
                                    db_dataset_id = db.store_dataset(
                                        f"HEALED_API_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                        healed_df,
                                        None,
                                        validation_report.get('final_quality_score', 0)
                                    )
                                    st.success(f"✅ Healed data saved to database! Dataset ID: {db_dataset_id}")
                                except Exception as e:
                                    st.error(f"❌ Failed to save to database: {str(e)}")
        else:
            st.info("💡 No API data available for healing. Fetch data from an API source above to see healing options.")

elif page == "⚙️ Settings & Health":
    st.title("⚙️ Settings & Health")
    
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
    
    # MS SQL Database Ingestion
    st.subheader("💾 MS SQL Database Ingestion")
    st.markdown("Connect to external MS SQL databases and import tables for data quality analysis.")
    
    # Initialize session state for MS SQL connection
    if 'mssql_connection_string' not in st.session_state:
        st.session_state['mssql_connection_string'] = ''
    if 'mssql_connected' not in st.session_state:
        st.session_state['mssql_connected'] = False
    if 'mssql_tables' not in st.session_state:
        st.session_state['mssql_tables'] = None
    
    # Connection string input
    conn_string = st.text_input(
        "External MS SQL Connection String",
        value=st.session_state['mssql_connection_string'],
        type="password",
        placeholder="DRIVER={ODBC Driver 17 for SQL Server};SERVER=your_server;DATABASE=your_db;UID=user;PWD=pass",
        help="Enter ODBC connection string for the external MS SQL database"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔌 Test Connection"):
            if not conn_string:
                st.error("Please enter a connection string")
            else:
                with st.spinner("Testing connection..."):
                    result = db.connect_external_mssql(conn_string)
                    
                    if result['success']:
                        st.success(result['message'])
                        st.session_state['mssql_connection_string'] = conn_string
                        st.session_state['mssql_connected'] = True
                        
                        # Auto-load tables after successful connection
                        try:
                            tables_df = db.list_external_tables(conn_string)
                            st.session_state['mssql_tables'] = tables_df
                        except Exception as e:
                            st.warning(f"Connected, but couldn't list tables: {str(e)}")
                    else:
                        st.error(result['message'])
                        st.session_state['mssql_connected'] = False
                        st.session_state['mssql_tables'] = None
    
    with col2:
        if st.session_state['mssql_connected'] and st.button("🔄 Refresh Tables"):
            with st.spinner("Loading tables..."):
                try:
                    tables_df = db.list_external_tables(st.session_state['mssql_connection_string'])
                    st.session_state['mssql_tables'] = tables_df
                    st.success(f"Found {len(tables_df)} tables")
                except Exception as e:
                    st.error(f"Failed to list tables: {str(e)}")
                    st.session_state['mssql_tables'] = None
    
    # Display tables if connected
    if st.session_state['mssql_connected'] and st.session_state['mssql_tables'] is not None:
        tables_df = st.session_state['mssql_tables']
        
        if not tables_df.empty:
            st.markdown("#### 📊 Available Tables")
            
            # Display tables in a nice format
            st.dataframe(
                tables_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Table import section
            st.markdown("#### 📥 Import Table")
            
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                selected_schema = st.selectbox(
                    "Schema",
                    options=sorted(tables_df['SchemaName'].unique().tolist())
                )
            
            with col2:
                # Filter tables by selected schema
                schema_tables = tables_df[tables_df['SchemaName'] == selected_schema]['TableName'].tolist()
                selected_table = st.selectbox(
                    "Table",
                    options=schema_tables
                )
            
            with col3:
                dataset_name = st.text_input(
                    "Dataset Name (optional)",
                    placeholder="Leave blank for auto-generated name"
                )
            
            if st.button("📥 Import Table", type="primary"):
                if selected_schema and selected_table:
                    with st.spinner(f"Importing {selected_schema}.{selected_table}..."):
                        result = db.import_table_from_mssql(
                            st.session_state['mssql_connection_string'],
                            selected_schema,
                            selected_table,
                            dataset_name if dataset_name else None
                        )
                        
                        if result['success']:
                            st.success(result['message'])
                            st.info(f"✅ Dataset ID: {result['dataset_id']} | Name: {result['dataset_name']}")
                            st.info("💡 The imported data is now available in the 'Upload & Process Data' page with status 'PENDING'. You can process it through the AutoHeal pipeline.")
                        else:
                            st.error(result['message'])
                else:
                    st.error("Please select both schema and table")
        else:
            st.info("No user tables found in the external database")
    
    st.markdown("---")
    
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

    # Agent Learning Insights
    st.subheader("🧠 Agent Learning Insights")
    
    if orchestrator and hasattr(orchestrator, 'learning_system'):
        learning_system = orchestrator.learning_system
        
        # Get learning insights for each agent
        agents = ['DataQualityAgent', 'DataHealingAgent', 'PipelineOrchestrator']
        
        for agent_name in agents:
            with st.expander(f"📊 {agent_name} Learning History", expanded=False):
                try:
                    # Get agent recommendations (includes learning insights)
                    insights = learning_system.get_agent_recommendations(agent_name)
                    
                    if insights and 'error' not in insights:
                        # Display learning metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            confidence = insights.get('overall_confidence', 0)
                            st.metric("Learning Confidence", f"{confidence:.1%}")
                        
                        with col2:
                            patterns_count = insights.get('learned_patterns_count', 0)
                            st.metric("Learned Patterns", patterns_count)
                        
                        with col3:
                            feedback_count = insights.get('human_feedback_insights', {}).get('feedback_count', 0)
                            st.metric("Human Feedback", feedback_count)
                        
                        # Display rejection pattern insights
                        if 'rejection_pattern_insights' in insights:
                            rejection_data = insights['rejection_pattern_insights']
                            
                            if rejection_data.get('total_rejections', 0) > 0:
                                st.markdown("#### 🚫 Rejection Patterns")
                                
                                # Most rejected actions
                                if rejection_data.get('rejection_patterns'):
                                    st.write("**Most Rejected Actions:**")
                                    for action, count in sorted(rejection_data['rejection_patterns'].items(), 
                                                              key=lambda x: x[1], reverse=True)[:3]:
                                        st.write(f"- {action.replace('_', ' ').title()}: {count} times")
                                
                                # Preferred alternatives
                                if rejection_data.get('preferred_alternatives'):
                                    st.write("**Preferred Alternatives:**")
                                    for alt, count in sorted(rejection_data['preferred_alternatives'].items(), 
                                                           key=lambda x: x[1], reverse=True)[:3]:
                                        st.write(f"- {alt}: {count} times")
                                
                                # Learning insights
                                if rejection_data.get('learning_insights'):
                                    st.write("**Key Learnings:**")
                                    for insight in rejection_data['learning_insights']:
                                        st.info(f"💡 {insight}")
                        
                        # Display human feedback insights
                        if 'human_feedback_insights' in insights:
                            feedback_data = insights['human_feedback_insights']
                            
                            if feedback_data.get('overall_satisfaction', 0) > 0:
                                st.markdown("#### 👤 Human Feedback Summary")
                                satisfaction = feedback_data['overall_satisfaction']
                                st.metric("Average Satisfaction", f"{satisfaction:.1f}/5.0")
                                
                                if feedback_data.get('improvement_areas'):
                                    st.write("**Areas for Improvement:**")
                                    for area in feedback_data['improvement_areas']:
                                        st.write(f"- {area.replace('_', ' ').title()}")
                        
                        # Display successful patterns
                        if insights.get('learned_patterns_count', 0) > 0:
                            st.markdown("#### ✅ Successful Patterns")
                            st.write(f"Agent has learned {insights['learned_patterns_count']} successful patterns from past operations.")
                            
                            # Show preferred alternatives if available
                            if 'human_preferred_alternatives' in insights:
                                st.write("**Human-Preferred Methods:**")
                                for method, count in insights['human_preferred_alternatives'].items():
                                    st.write(f"- {method}: Preferred {count} times")
                    
                    else:
                        st.info(f"No learning data available for {agent_name} yet. The agent will start learning from your interactions.")
                        
                except Exception as e:
                    st.error(f"Error loading learning insights for {agent_name}: {str(e)}")
    
    else:
        st.warning("⚠️ Learning system not available. Agent learning features are disabled.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🩺 AutoHeal v1.0**")
st.sidebar.markdown("*AI Data Pipeline*")