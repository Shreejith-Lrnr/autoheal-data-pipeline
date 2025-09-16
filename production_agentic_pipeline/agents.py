import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from config import Config
from database import ProductionDatabase

class ProductionAIAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.db = ProductionDatabase()
        
    def call_groq_api(self, prompt, temperature=0.1):
        """Production-ready API call with error handling and fallback"""
        headers = {
            "Authorization": f"Bearer {Config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": Config.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        try:
            # Try with SSL verification first
            response = requests.post(Config.GROQ_API_URL, headers=headers, json=data, timeout=30)
            
            # Debug: Print response details for 400 errors
            if response.status_code == 400:
                print(f"400 Bad Request Details:")
                print(f"Response: {response.text}")
                print(f"Request headers: {headers}")
                print(f"Request data: {data}")
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            if "400" in str(e):
                print(f"HTTP 400 Error: {e}")
                print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
                # Try with different model as fallback
                return self._try_alternative_model(prompt, temperature, headers)
            else:
                print(f"HTTP Error: {e}")
                return self._generate_fallback_response(prompt)
                
        except requests.exceptions.SSLError:
            try:
                # Handle corporate SSL issues
                response = requests.post(Config.GROQ_API_URL, headers=headers, json=data, timeout=30, verify=False)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"SSL fallback failed: {e}")
                return self._generate_fallback_response(prompt)
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error to Groq API: {e}")
            return self._generate_fallback_response(prompt)
        except requests.exceptions.Timeout:
            print("Groq API timeout - using fallback")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            print(f"Groq API error: {e}")
            return self._generate_fallback_response(prompt)
    
    def _try_alternative_model(self, prompt, temperature, headers):
        """Try alternative models when the default fails"""
        alternative_models = [
            "llama3-70b-8192",  # Larger model
            "mixtral-8x7b-32768",  # Different architecture
            "gemma-7b-it"  # Smaller model
        ]
        
        for model in alternative_models:
            try:
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 1000
                }
                
                response = requests.post(Config.GROQ_API_URL, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    print(f"Success with alternative model: {model}")
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"Alternative model {model} also failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Alternative model {model} error: {e}")
                continue
        
        print("All alternative models failed, using fallback")
        return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt):
        """Generate fallback response when API is unavailable"""
        if "executive summary" in prompt.lower():
            return """
            EXECUTIVE SUMMARY (Offline Mode):
            Data quality analysis completed using local algorithms. The system detected several data quality issues that require attention. Automated healing suggestions have been generated based on standard data cleaning best practices.
            
            BUSINESS IMPACT:
            Data quality issues may affect analytical accuracy and decision-making. Recommended to address high-priority issues first to maintain data integrity.
            
            RECOMMENDED NEXT STEPS:
            1. Review and approve suggested healing actions
            2. Monitor data quality improvements after healing
            3. Implement data validation rules to prevent future issues
            
            RISK ASSESSMENT:
            Medium risk - Issues are common and addressable with standard cleaning procedures.
            """
        elif "recommendations" in prompt.lower():
            return """
            OFFLINE AI RECOMMENDATIONS:
            1. [PRIORITY: HIGH] - Address null values using appropriate imputation methods
            2. [PRIORITY: MEDIUM] - Remove or handle duplicate records to improve data consistency  
            3. [PRIORITY: LOW] - Review outliers for potential data entry errors or valid extreme values
            """
        elif "execution order" in prompt.lower():
            return """
            EXECUTION_ORDER: HANDLE_NULLS, REMOVE_DUPLICATES, HANDLE_OUTLIERS
            REASONING: Handle missing values first to ensure completeness, then remove duplicates to avoid processing redundant data, finally address outliers which may be valid extreme values requiring careful consideration.
            """
        else:
            return "AI analysis completed in offline mode. Standard data quality recommendations applied based on detected issues."

class DataQualityAgent(ProductionAIAgent):
    def __init__(self):
        super().__init__("DataQualityAgent", "Data Quality Specialist")
    
    def analyze_dataset(self, df, dataset_name):
        """Comprehensive data quality analysis"""
        quality_report = {
            'dataset_name': dataset_name,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'issues': [],
            'quality_score': 0,
            'recommendations': []
        }
        
        # Check for null values
        null_analysis = df.isnull().sum()
        for col, null_count in null_analysis.items():
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                severity = "HIGH" if null_percentage > 10 else "MEDIUM" if null_percentage > 5 else "LOW"
                quality_report['issues'].append({
                    'type': 'NULL_VALUES',
                    'column': col,
                    'affected_rows': int(null_count),
                    'percentage': round(null_percentage, 2),
                    'severity': severity
                })
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            severity = "HIGH" if duplicate_percentage > 5 else "MEDIUM" if duplicate_percentage > 2 else "LOW"
            quality_report['issues'].append({
                'type': 'DUPLICATES',
                'affected_rows': int(duplicate_count),
                'percentage': round(duplicate_percentage, 2),
                'severity': severity
            })
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if not df[col].empty:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > Config.QUALITY_RULES['outlier_std_threshold']).sum()
                if outliers > 0:
                    outlier_percentage = (outliers / len(df)) * 100
                    severity = "HIGH" if outlier_percentage > 5 else "MEDIUM" if outlier_percentage > 2 else "LOW"
                    quality_report['issues'].append({
                        'type': 'OUTLIERS',
                        'column': col,
                        'affected_rows': int(outliers),
                        'percentage': round(outlier_percentage, 2),
                        'severity': severity
                    })
        
        # Calculate quality score
        total_issues = sum(issue['affected_rows'] for issue in quality_report['issues'])
        quality_report['quality_score'] = max(0, 100 - (total_issues / len(df)) * 100)
        
        # Get AI recommendations
        if quality_report['issues']:
            ai_prompt = f"""
            As a data quality expert, analyze these issues and provide specific recommendations:
            
            Dataset: {dataset_name} ({len(df)} records, {len(df.columns)} columns)
            Issues found: {json.dumps(quality_report['issues'], indent=2)}
            
            Provide 3 specific, actionable recommendations in this format:
            1. [PRIORITY: HIGH/MEDIUM/LOW] - [Specific action]
            2. [PRIORITY: HIGH/MEDIUM/LOW] - [Specific action]  
            3. [PRIORITY: HIGH/MEDIUM/LOW] - [Specific action]
            """
            
            ai_recommendations = self.call_groq_api(ai_prompt)
            quality_report['ai_analysis'] = ai_recommendations
        
        return quality_report

class DataHealingAgent(ProductionAIAgent):
    def __init__(self):
        super().__init__("DataHealingAgent", "Data Healing Specialist")
    
    def propose_healing_actions(self, df, quality_report):
        """Propose specific healing actions for data issues"""
        healing_plan = {
            'actions': [],
            'estimated_impact': {},
            'execution_order': []
        }
        
        for issue in quality_report['issues']:
            if issue['type'] == 'NULL_VALUES' and issue['severity'] in ['HIGH', 'MEDIUM']:
                action = {
                    'type': 'HANDLE_NULLS',
                    'column': issue['column'],
                    'affected_rows': issue['affected_rows'],
                    'options': []
                }
                
                # Determine best null handling strategy
                col_data = df[issue['column']]
                if col_data.dtype in ['int64', 'float64']:
                    action['options'] = [
                        {'method': 'FILL_MEAN', 'description': f"Fill with mean value ({col_data.mean():.2f})"},
                        {'method': 'FILL_MEDIAN', 'description': f"Fill with median value ({col_data.median():.2f})"},
                        {'method': 'DELETE_ROWS', 'description': f"Delete {issue['affected_rows']} rows with null values"}
                    ]
                else:
                    action['options'] = [
                        {'method': 'FILL_MODE', 'description': f"Fill with most common value"},
                        {'method': 'FILL_DEFAULT', 'description': f"Fill with default value 'Unknown'"},
                        {'method': 'DELETE_ROWS', 'description': f"Delete {issue['affected_rows']} rows with null values"}
                    ]
                
                healing_plan['actions'].append(action)
            
            elif issue['type'] == 'DUPLICATES' and issue['severity'] in ['HIGH', 'MEDIUM']:
                healing_plan['actions'].append({
                    'type': 'REMOVE_DUPLICATES',
                    'affected_rows': issue['affected_rows'],
                    'options': [
                        {'method': 'KEEP_FIRST', 'description': 'Keep first occurrence of duplicates'},
                        {'method': 'KEEP_LAST', 'description': 'Keep last occurrence of duplicates'}
                    ]
                })
            
            elif issue['type'] == 'OUTLIERS' and issue['severity'] == 'HIGH':
                healing_plan['actions'].append({
                    'type': 'HANDLE_OUTLIERS',
                    'column': issue['column'],
                    'affected_rows': issue['affected_rows'],
                    'options': [
                        {'method': 'CAP_OUTLIERS', 'description': 'Cap outliers to 95th percentile'},
                        {'method': 'REMOVE_OUTLIERS', 'description': 'Remove outlier records'},
                        {'method': 'LOG_TRANSFORM', 'description': 'Apply log transformation'}
                    ]
                })
        
        # Get AI recommendations for execution order
        if healing_plan['actions']:
            ai_prompt = f"""
            As a data healing specialist, recommend the optimal execution order for these data cleaning actions:
            
            Actions to perform: {json.dumps(healing_plan['actions'], indent=2)}
            
            Consider:
            1. Which actions should be performed first to minimize data loss
            2. Dependencies between actions
            3. Impact on data integrity
            
            Respond with:
            EXECUTION_ORDER: [comma-separated list of action types in order]
            REASONING: [brief explanation of the order]
            """
            
            ai_order = self.call_groq_api(ai_prompt)
            healing_plan['ai_execution_plan'] = ai_order
        
        return healing_plan
    
    def execute_healing_action(self, df, action, method):
        """Execute a specific healing action"""
        cleaned_df = df.copy()
        execution_log = {
            'action': action,
            'method': method,
            'original_shape': df.shape,
            'success': False,
            'message': ''
        }
        
        try:
            if action['type'] == 'HANDLE_NULLS':
                col = action['column']
                if method == 'FILL_MEAN':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif method == 'FILL_MEDIAN':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif method == 'FILL_MODE':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                elif method == 'FILL_DEFAULT':
                    cleaned_df[col].fillna('Unknown', inplace=True)
                elif method == 'DELETE_ROWS':
                    cleaned_df = cleaned_df.dropna(subset=[col])
                
            elif action['type'] == 'REMOVE_DUPLICATES':
                if method == 'KEEP_FIRST':
                    cleaned_df = cleaned_df.drop_duplicates(keep='first')
                elif method == 'KEEP_LAST':
                    cleaned_df = cleaned_df.drop_duplicates(keep='last')
            
            elif action['type'] == 'HANDLE_OUTLIERS':
                col = action['column']
                if method == 'CAP_OUTLIERS':
                    percentile_95 = cleaned_df[col].quantile(0.95)
                    cleaned_df[col] = cleaned_df[col].clip(upper=percentile_95)
                elif method == 'REMOVE_OUTLIERS':
                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                    cleaned_df = cleaned_df[z_scores <= Config.QUALITY_RULES['outlier_std_threshold']]
                elif method == 'LOG_TRANSFORM':
                    cleaned_df[col] = np.log1p(cleaned_df[col])
            
            execution_log['final_shape'] = cleaned_df.shape
            execution_log['records_affected'] = df.shape[0] - cleaned_df.shape[0]
            execution_log['success'] = True
            execution_log['message'] = f"Successfully executed {action['type']} using {method}"
            
        except Exception as e:
            execution_log['message'] = f"Execution failed: {str(e)}"
        
        return cleaned_df, execution_log
    
    def recalculate_quality_score(self, df):
        """Recalculate quality score after healing"""
        total_issues = 0
        
        # Count null values
        total_issues += df.isnull().sum().sum()
        
        # Count duplicates
        total_issues += df.duplicated().sum()
        
        # Count outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if not df[col].empty and len(df[col].dropna()) > 1:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                total_issues += (z_scores > Config.QUALITY_RULES['outlier_std_threshold']).sum()
        
        # Calculate quality score
        quality_score = max(0, 100 - (total_issues / len(df)) * 100)
        return quality_score

class PipelineOrchestratorAgent(ProductionAIAgent):
    def __init__(self):
        super().__init__("PipelineOrchestrator", "Pipeline Coordination Specialist")
        self.quality_agent = DataQualityAgent()
        self.healing_agent = DataHealingAgent()
    
    def process_dataset(self, df, dataset_name):
        """Orchestrate complete dataset processing"""
        processing_log = {
            'start_time': datetime.now(),
            'dataset_name': dataset_name,
            'original_shape': df.shape,
            'steps_completed': [],
            'human_approvals_needed': [],
            'final_recommendations': []
        }
        
        # Store dataset in database
        dataset_id = self.db.store_dataset(dataset_name, df)
        
        # LOG ACTION: Processing Start
        start_action_id = self.db.log_agent_action(
            dataset_id, self.name, 
            "DATASET_PROCESSING_START", 
            f"Starting pipeline processing for {dataset_name} ({len(df)} records, {len(df.columns)} columns)",
            "EXECUTING"
        )
        
        # LOG ACTION: Quality Analysis Start
        quality_action_id = self.db.log_agent_action(
            dataset_id, self.quality_agent.name, 
            "QUALITY_ANALYSIS_START", 
            f"Beginning comprehensive data quality analysis",
            "EXECUTING"
        )
        
        # Step 1: Quality Analysis
        quality_report = self.quality_agent.analyze_dataset(df, dataset_name)
        processing_log['steps_completed'].append("Quality Analysis")
        
        # Update dataset with actual quality score
        self.db.update_dataset_quality_score(dataset_id, quality_report['quality_score'])
        
        # Update quality analysis completion
        self.db.update_agent_action(quality_action_id, "COMPLETED")
        
        # LOG ACTION: Quality Analysis Results
        self.db.log_agent_action(
            dataset_id, self.quality_agent.name,
            "QUALITY_ANALYSIS_COMPLETE",
            f"Analysis complete - Quality Score: {quality_report['quality_score']:.1f}/100, Issues Found: {len(quality_report['issues'])}",
            "COMPLETED"
        )
        
        # Log quality issues to database
        for issue in quality_report['issues']:
            # Log each individual issue as an agent action
            self.db.log_agent_action(
                dataset_id, self.quality_agent.name,
                f"ISSUE_DETECTED_{issue['type']}",
                f"Column '{issue.get('column', 'Multiple')}': {issue['affected_rows']} rows affected ({issue['percentage']}%) - Severity: {issue['severity']}",
                "COMPLETED"
            )
            
            self.db.log_quality_issue(
                dataset_id, issue['type'], 
                f"{issue.get('column', 'Multiple columns')}: {issue['percentage']}% affected",
                issue['affected_rows'], issue['severity']
            )
        
        # Step 2: Healing Plan
        if quality_report['issues']:
            healing_action_id = self.db.log_agent_action(
                dataset_id, self.healing_agent.name,
                "HEALING_PLAN_START",
                f"Generating healing plan for {len(quality_report['issues'])} detected issues",
                "EXECUTING"
            )
            
            healing_plan = self.healing_agent.propose_healing_actions(df, quality_report)
            processing_log['steps_completed'].append("Healing Plan Generation")
            
            # Update healing plan completion
            self.db.update_agent_action(healing_action_id, "COMPLETED")
            
            # LOG ACTION: Healing Plan Results
            self.db.log_agent_action(
                dataset_id, self.healing_agent.name,
                "HEALING_PLAN_COMPLETE",
                f"Generated {len(healing_plan['actions'])} healing actions: {', '.join([action['type'] for action in healing_plan['actions']])}",
                "COMPLETED"
            )
            
            # Log each proposed healing action
            for i, action in enumerate(healing_plan['actions']):
                action_id = self.db.log_agent_action(
                    dataset_id, self.healing_agent.name,
                    f"HEALING_PROPOSED_{action['type']}",
                    f"Proposed {action['type']} for column '{action.get('column', 'multiple')}' affecting {action.get('affected_rows', 0)} records",
                    "PENDING_APPROVAL"
                )
                
                # Determine which actions need human approval
                if action.get('affected_rows', 0) > len(df) * 0.1:  # > 10% of data
                    processing_log['human_approvals_needed'].append({
                        'action_id': action_id,
                        'action_type': action['type'],
                        'impact': f"Affects {action.get('affected_rows', 0)} records",
                        'recommendation': action['options'][0] if action.get('options') else None
                    })
                else:
                    # Auto-approve low impact actions
                    self.db.update_agent_action(action_id, "AUTO_APPROVED", "AUTO_APPROVED")
                    self.db.log_agent_action(
                        dataset_id, self.name,
                        f"AUTO_APPROVED_{action['type']}",
                        f"Auto-approved low-impact action affecting {action.get('affected_rows', 0)} records (<10% threshold)",
                        "COMPLETED"
                    )
        else:
            # Log no issues found
            self.db.log_agent_action(
                dataset_id, self.quality_agent.name,
                "NO_ISSUES_FOUND",
                "Data quality analysis completed - no significant issues detected. Dataset is healthy!",
                "COMPLETED"
            )
            healing_plan = {'actions': [], 'ai_execution_plan': 'No healing needed'}
        
        # Final AI recommendations
        if quality_report['issues'] or processing_log['human_approvals_needed']:
            summary_action_id = self.db.log_agent_action(
                dataset_id, self.name,
                "AI_SUMMARY_START",
                "Generating executive summary and business impact assessment",
                "EXECUTING"
            )
            
            ai_summary_prompt = f"""
            As a Pipeline Orchestrator, provide executive summary and recommendations:
            
            Dataset: {dataset_name}
            Quality Score: {quality_report['quality_score']:.1f}/100
            Issues Found: {len(quality_report['issues'])}
            Actions Requiring Approval: {len(processing_log['human_approvals_needed'])}
            
            Quality Issues:
            {json.dumps(quality_report['issues'], indent=2)}
            
            Provide:
            1. Executive summary (2-3 sentences)
            2. Business impact assessment
            3. Recommended next steps
            4. Risk assessment
            """
            
            ai_summary = self.call_groq_api(ai_summary_prompt)
            processing_log['ai_executive_summary'] = ai_summary
            
            # Update summary completion
            self.db.update_agent_action(summary_action_id, "COMPLETED")
            
            # LOG ACTION: Summary Complete
            self.db.log_agent_action(
                dataset_id, self.name,
                "AI_SUMMARY_COMPLETE",
                f"Executive summary generated - {len(processing_log['human_approvals_needed'])} actions require human approval",
                "COMPLETED"
            )
        
        # Update processing start action
        self.db.update_agent_action(start_action_id, "COMPLETED")
        
        # Update final dataset status
        final_status = "COMPLETED" if not processing_log['human_approvals_needed'] else "PENDING_APPROVAL"
        self.db.update_dataset_quality_score(dataset_id, quality_report['quality_score'], status=final_status)
        
        # LOG ACTION: Processing Complete
        self.db.log_agent_action(
            dataset_id, self.name,
            "DATASET_PROCESSING_COMPLETE",
            f"Pipeline processing complete for {dataset_name} - Status: {final_status}, Quality Score: {quality_report['quality_score']:.1f}/100",
            "COMPLETED"
        )
        
        processing_log['end_time'] = datetime.now()
        processing_log['quality_report'] = quality_report
        processing_log['healing_plan'] = healing_plan
        
        return processing_log, dataset_id
