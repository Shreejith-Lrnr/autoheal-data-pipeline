import requests
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import Config
from database import ProductionDatabase
from agent_learning import AgentLearningSystem

class ProductionAIAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.db = ProductionDatabase()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.logger.setLevel(logging.INFO)
        
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
            "llama-3.1-70b-versatile",  # Larger model
            "llama-3.1-405b-instruct",  # Largest model
            "llama3-70b-8192",          # Fallback
            "gemma2-9b-it"               # Alternative
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
        self.learning_system = AgentLearningSystem(self.db)
        print("DataQualityAgent initialized with learning system")
    
    def analyze_dataset_optimized(self, df, dataset_name, sample_size=50000, max_problematic_rows=10000):
        """Optimized dataset analysis for large datasets - only analyzes problematic rows"""

        # Determine if we need sampling
        use_sampling = len(df) > sample_size
        analysis_sample_size = min(sample_size, len(df))

        # Initialize quality report
        quality_report = {
            'dataset_name': dataset_name,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'issues': [],
            'quality_score': 0,
            'recommendations': [],
            'analysis_method': 'selective_rows',
            'sampled_records': analysis_sample_size if use_sampling else len(df),
            'problematic_rows_analyzed': 0
        }

        try:
            # Step 1: Identify problematic rows efficiently
            problematic_indices = self._identify_problematic_rows(df, max_problematic_rows)

            quality_report['problematic_rows_identified'] = len(problematic_indices)

            # Step 2: Analyze only problematic rows (or sample if too many)
            if len(problematic_indices) > 0:
                # Limit analysis to prevent memory issues
                analysis_indices = problematic_indices[:max_problematic_rows]
                problematic_df = df.iloc[analysis_indices]

                quality_report['problematic_rows_analyzed'] = len(analysis_indices)

                # Analyze null values in problematic rows only
                null_analysis = problematic_df.isnull().sum()
                for col, null_count in null_analysis.items():
                    if null_count > 0:
                        # Calculate percentage based on total dataset, not just problematic rows
                        total_nulls_in_col = df[col].isnull().sum()
                        null_percentage = (total_nulls_in_col / len(df)) * 100

                        severity = ("HIGH" if null_percentage > 15
                                  else "MEDIUM" if null_percentage > 5
                                  else "LOW")

                        quality_report['issues'].append({
                            'type': 'NULL_VALUES',
                            'column': col,
                            'affected_rows': int(total_nulls_in_col),
                            'percentage': round(null_percentage, 2),
                            'severity': severity,
                            'analysis_method': 'selective_rows'
                        })

            # Step 3: Check for duplicates (efficient sampling approach)
            if use_sampling:
                # Sample for duplicate detection to avoid full scan
                sample_df = df.sample(n=analysis_sample_size, random_state=42)
                sample_duplicates = sample_df.duplicated().sum()
                if sample_duplicates > 0:
                    # Estimate total duplicates
                    estimated_duplicates = int((sample_duplicates / analysis_sample_size) * len(df))
                    duplicate_percentage = (estimated_duplicates / len(df)) * 100

                    severity = ("HIGH" if duplicate_percentage > 10
                              else "MEDIUM" if duplicate_percentage > 2
                              else "LOW")

                    quality_report['issues'].append({
                        'type': 'DUPLICATES_ESTIMATED',
                        'affected_rows': estimated_duplicates,
                        'percentage': round(duplicate_percentage, 2),
                        'severity': severity,
                        'analysis_method': 'sampled_estimation'
                    })
            else:
                # Full scan for smaller datasets
                duplicate_count = df.duplicated().sum()
                if duplicate_count > 0:
                    duplicate_percentage = (duplicate_count / len(df)) * 100
                    severity = ("HIGH" if duplicate_percentage > 10
                              else "MEDIUM" if duplicate_percentage > 2
                              else "LOW")

                    quality_report['issues'].append({
                        'type': 'DUPLICATES',
                        'affected_rows': int(duplicate_count),
                        'percentage': round(duplicate_percentage, 2),
                        'severity': severity,
                        'analysis_method': 'full_scan'
                    })

            # Step 4: Check for outliers in numeric columns (efficient approach)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if use_sampling:
                    # Use statistical bounds instead of full z-score calculation
                    sample_values = df[col].dropna().sample(n=min(analysis_sample_size, len(df[col].dropna())), random_state=42)
                    if len(sample_values) > 0:
                        Q1 = sample_values.quantile(0.25)
                        Q3 = sample_values.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        # Estimate outliers in full dataset
                        estimated_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        outlier_percentage = (estimated_outliers / len(df)) * 100

                        if estimated_outliers > 0:
                            severity = ("HIGH" if outlier_percentage > 10
                                      else "MEDIUM" if outlier_percentage > 3
                                      else "LOW")

                            quality_report['issues'].append({
                                'type': 'OUTLIERS_ESTIMATED',
                                'column': col,
                                'affected_rows': int(estimated_outliers),
                                'percentage': round(outlier_percentage, 2),
                                'severity': severity,
                                'analysis_method': 'statistical_estimation'
                            })
                else:
                    # Full analysis for smaller datasets
                    if not df[col].empty:
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        outliers = (z_scores > Config.QUALITY_RULES['outlier_std_threshold']).sum()
                        if outliers > 0:
                            outlier_percentage = (outliers / len(df)) * 100
                            severity = ("HIGH" if outlier_percentage > 10
                                      else "MEDIUM" if outlier_percentage > 3
                                      else "LOW")

                            quality_report['issues'].append({
                                'type': 'OUTLIERS',
                                'column': col,
                                'affected_rows': int(outliers),
                                'percentage': round(outlier_percentage, 2),
                                'severity': severity,
                                'analysis_method': 'full_scan'
                            })

            # Step 5: Calculate quality score
            total_issues = sum(issue['affected_rows'] for issue in quality_report['issues'])
            quality_report['quality_score'] = max(0, 100 - (total_issues / len(df)) * 100)

            # Step 6: Generate recommendations
            if quality_report['issues']:
                quality_report['recommendations'] = self._generate_optimized_recommendations(quality_report, len(df))

            return quality_report

        except Exception as e:
            self.logger.error(f"Error in optimized dataset analysis: {str(e)}")
            # Fallback to basic analysis
            return self.analyze_dataset_basic(df, dataset_name)

    def analyze_dataset(self, df, dataset_name):
        """Main dataset analysis method - automatically chooses optimized or basic analysis"""
        # Use optimized analysis for large datasets (>100k rows) or datasets with many columns
        if len(df) > 100000 or len(df.columns) > 50:
            return self.analyze_dataset_optimized(df, dataset_name)
        else:
            return self.analyze_dataset_basic(df, dataset_name)

    def _identify_problematic_rows(self, df, max_rows=10000):
        """Efficiently identify rows that need detailed analysis"""
        problematic_indices = set()

        # Find rows with any null values
        null_rows = df[df.isnull().any(axis=1)].index
        problematic_indices.update(null_rows[:max_rows])  # Limit to prevent memory issues

        # Find duplicate rows (limit to prevent full scan)
        if len(df) > 100000:
            # For very large datasets, sample for duplicates
            sample_size = min(50000, len(df))
            sample_indices = df.sample(n=sample_size, random_state=42).index
            sample_duplicates = df.loc[sample_indices][df.loc[sample_indices].duplicated()].index
            problematic_indices.update(sample_duplicates[:max_rows//2])
        else:
            duplicate_rows = df[df.duplicated()].index
            problematic_indices.update(duplicate_rows[:max_rows//2])

        # Find rows with outlier values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 1000:  # Only check if enough data
                try:
                    # Use IQR method for efficiency
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                    problematic_indices.update(outlier_rows[:max_rows//4])  # Limit per column
                except:
                    continue  # Skip if calculation fails

        return list(problematic_indices)[:max_rows]  # Final limit

    def _generate_optimized_recommendations(self, quality_report, total_rows):
        """Generate recommendations based on optimized analysis"""
        recommendations = []

        for issue in quality_report['issues']:
            if issue['type'] == 'NULL_VALUES':
                if issue['severity'] == 'HIGH':
                    recommendations.append(f"Critical: Fill or remove {issue['affected_rows']} null values in {issue['column']} ({issue['percentage']}%)")
                else:
                    recommendations.append(f"Consider filling {issue['affected_rows']} null values in {issue['column']}")

            elif 'DUPLICATES' in issue['type']:
                if issue['severity'] == 'HIGH':
                    recommendations.append(f"Critical: Remove {issue['affected_rows']} duplicate rows ({issue['percentage']}%)")
                else:
                    recommendations.append(f"Review {issue['affected_rows']} potential duplicate rows")

            elif 'OUTLIERS' in issue['type']:
                recommendations.append(f"Review {issue['affected_rows']} outlier values in {issue['column']} ({issue['percentage']}%)")

        return recommendations

    def analyze_dataset_basic(self, df, dataset_name):
        """Comprehensive data quality analysis with learning integration"""
        start_time = datetime.now()
        
        # Get learning insights for this context
        learning_insights = self.learning_system.get_agent_recommendations(
            "DataQualityAgent", 
            f"quality_analysis_{dataset_name}"
        )
        
        # Apply learned thresholds and preferences
        learned_thresholds = learning_insights.get('severity_thresholds', {
            'null_high': 10, 'null_medium': 5,
            'duplicate_high': 5, 'duplicate_medium': 2,
            'outlier_high': 5, 'outlier_medium': 2
        })
        
        quality_report = {
            'dataset_name': dataset_name,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'issues': [],
            'quality_score': 0,
            'recommendations': [],
            'learning_applied': True,
            'learned_thresholds': learned_thresholds,
            'learning_confidence': learning_insights.get('overall_confidence', 0.5)
        }
        
        success = True
        error_message = None
        
        try:
            # Check for null values with learned thresholds
            null_analysis = df.isnull().sum()
            for col, null_count in null_analysis.items():
                if null_count > 0:
                    null_percentage = (null_count / len(df)) * 100
                    # Use learned thresholds instead of fixed values
                    severity = ("HIGH" if null_percentage > learned_thresholds['null_high'] 
                              else "MEDIUM" if null_percentage > learned_thresholds['null_medium'] 
                              else "LOW")
                    
                    quality_report['issues'].append({
                        'type': 'NULL_VALUES',
                        'column': col,
                        'affected_rows': int(null_count),
                        'percentage': round(null_percentage, 2),
                        'severity': severity,
                        'learned_threshold_used': True
                    })
        
            # Check for duplicates with learned thresholds
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                duplicate_percentage = (duplicate_count / len(df)) * 100
                severity = ("HIGH" if duplicate_percentage > learned_thresholds['duplicate_high'] 
                          else "MEDIUM" if duplicate_percentage > learned_thresholds['duplicate_medium'] 
                          else "LOW")
                
                quality_report['issues'].append({
                    'type': 'DUPLICATES',
                    'affected_rows': int(duplicate_count),
                    'percentage': round(duplicate_percentage, 2),
                    'severity': severity,
                    'learned_threshold_used': True
                })
            
            # Check for outliers in numeric columns with learned thresholds
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if not df[col].empty:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = (z_scores > Config.QUALITY_RULES['outlier_std_threshold']).sum()
                    if outliers > 0:
                        outlier_percentage = (outliers / len(df)) * 100
                        severity = ("HIGH" if outlier_percentage > learned_thresholds['outlier_high'] 
                                  else "MEDIUM" if outlier_percentage > learned_thresholds['outlier_medium'] 
                                  else "LOW")
                        
                        quality_report['issues'].append({
                            'type': 'OUTLIERS',
                            'column': col,
                            'affected_rows': int(outliers),
                            'percentage': round(outlier_percentage, 2),
                            'severity': severity,
                            'learned_threshold_used': True
                        })
            
            # Calculate quality score with learned weights
            learned_weights = learning_insights.get('quality_weights', {
                'completeness': 0.4, 'consistency': 0.3, 'uniqueness': 0.3
            })
            
            total_issues = sum(issue['affected_rows'] for issue in quality_report['issues'])
            quality_report['quality_score'] = max(0, 100 - (total_issues / len(df)) * 100)
            
            # Get AI recommendations with learning context
            if quality_report['issues']:
                successful_patterns = learning_insights.get('successful_patterns', [])
                focus_areas = learning_insights.get('focus_areas', ['completeness', 'consistency'])
                
                ai_prompt = f"""
                As a data quality expert with learning from past analyses, analyze these issues:
                
                Dataset: {dataset_name} ({len(df)} records, {len(df.columns)} columns)
                Issues found: {json.dumps(quality_report['issues'], indent=2)}
                
                Learning Context:
                - Past successful approaches: {successful_patterns}
                - Recommended focus areas: {focus_areas}
                - Learning confidence: {learning_insights.get('overall_confidence', 0.5)}
                
                Provide 3 specific, actionable recommendations incorporating learned best practices:
                1. [PRIORITY: HIGH/MEDIUM/LOW] - [Specific action based on learning]
                2. [PRIORITY: HIGH/MEDIUM/LOW] - [Specific action based on learning]  
                3. [PRIORITY: HIGH/MEDIUM/LOW] - [Specific action based on learning]
                """
                
                ai_recommendations = self.call_groq_api(ai_prompt)
                quality_report['ai_analysis'] = ai_recommendations
            
            # Log successful analysis for learning
            execution_time = (datetime.now() - start_time).total_seconds()
            self.learning_system.log_agent_action(
                agent_name="DataQualityAgent",
                action_type="quality_analysis",
                context=f"dataset_{dataset_name}",
                input_data={
                    'dataset_size': [len(df), len(df.columns)],
                    'learning_confidence': learning_insights.get('overall_confidence', 0.5)
                },
                output_data={
                    'quality_score': quality_report['quality_score'],
                    'issues_found': len(quality_report['issues']),
                    'severity_breakdown': {
                        'high': len([i for i in quality_report['issues'] if i['severity'] == 'HIGH']),
                        'medium': len([i for i in quality_report['issues'] if i['severity'] == 'MEDIUM']),
                        'low': len([i for i in quality_report['issues'] if i['severity'] == 'LOW'])
                    }
                },
                success=success,
                execution_time=execution_time,
                error_message=error_message
            )
            
            print(f"✅ Quality analysis complete with learning - Score: {quality_report['quality_score']:.1f}%, Issues: {len(quality_report['issues'])}")
            
        except Exception as e:
            success = False
            error_message = str(e)
            quality_report['error'] = str(e)
            print(f"❌ Error in quality analysis: {str(e)}")
            
            # Log failed analysis for learning
            execution_time = (datetime.now() - start_time).total_seconds()
            self.learning_system.log_agent_action(
                agent_name="DataQualityAgent",
                action_type="quality_analysis",
                context=f"dataset_{dataset_name}",
                input_data={
                    'dataset_size': [len(df), len(df.columns)]
                },
                output_data={
                    'error': str(e)
                },
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
        
        return quality_report

class DataHealingAgent(ProductionAIAgent):
    def __init__(self):
        super().__init__("DataHealingAgent", "Data Healing Specialist")
        self.learning_system = AgentLearningSystem(self.db)
        print("DataHealingAgent initialized with learning system")
    
    def calculate_action_confidence(self, df, action, method, learning_insights):
        """Calculate confidence score for a healing action (0.0 - 1.0)"""
        confidence_factors = []
        
        # Factor 1: Data characteristics (40% weight)
        if action['type'] == 'HANDLE_NULLS':
            null_percentage = (action['affected_rows'] / len(df)) * 100
            
            # Low null % = higher confidence
            if null_percentage < 5:
                confidence_factors.append(0.95)  # Very confident for small % nulls
            elif null_percentage < 15:
                confidence_factors.append(0.80)
            elif null_percentage < 30:
                confidence_factors.append(0.60)
            else:
                confidence_factors.append(0.40)  # Low confidence for high % nulls
            
            # Numeric columns with mean/median are safer
            col_data = df[action['column']]
            if col_data.dtype in ['int64', 'float64'] and method in ['FILL_MEAN', 'FILL_MEDIAN']:
                confidence_factors.append(0.85)
            elif method == 'DELETE_ROWS' and null_percentage > 20:
                confidence_factors.append(0.50)  # Risky to delete many rows
            else:
                confidence_factors.append(0.70)
                
        elif action['type'] == 'REMOVE_DUPLICATES':
            dup_percentage = (action['affected_rows'] / len(df)) * 100
            
            # Removing duplicates is generally safe
            if dup_percentage < 10:
                confidence_factors.append(0.95)
            elif dup_percentage < 30:
                confidence_factors.append(0.75)
            else:
                confidence_factors.append(0.60)  # Many dupes might indicate data issue
                
        elif action['type'] == 'HANDLE_OUTLIERS':
            # Outlier handling is riskier
            if method == 'CAP_OUTLIERS':
                confidence_factors.append(0.75)
            elif method == 'REMOVE_OUTLIERS':
                confidence_factors.append(0.65)  # More risky
            else:
                confidence_factors.append(0.70)
        
        # Factor 2: Learning from past (30% weight)
        learned_confidence = learning_insights.get('overall_confidence', 0.5)
        preferred_methods = learning_insights.get('preferred_methods', {})
        
        # Boost confidence if method matches learned preferences
        if method in preferred_methods:
            learned_confidence = min(1.0, learned_confidence + 0.15)
        
        confidence_factors.append(learned_confidence)
        
        # Factor 3: Severity-based confidence (30% weight)
        severity_confidence = {
            'LOW': 0.90,     # Low severity = safe to auto-execute
            'MEDIUM': 0.70,  # Medium = might need approval
            'HIGH': 0.50     # High = definitely need human review
        }
        confidence_factors.append(severity_confidence.get(action.get('severity', 'MEDIUM'), 0.70))
        
        # Calculate weighted average
        final_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return round(final_confidence, 2)
    
    def generate_action_reasoning(self, df, action, method, confidence):
        """Generate chain-of-thought reasoning for action"""
        reasoning = {
            'decision': method,
            'confidence': confidence,
            'thought_process': [],
            'alternatives': [],
            'risks': []
        }
        
        # Thought process
        if action['type'] == 'HANDLE_NULLS':
            col_data = df[action['column']]
            null_pct = (action['affected_rows'] / len(df)) * 100
            
            reasoning['thought_process'].append(
                f"Column '{action['column']}' has {null_pct:.1f}% null values ({action['affected_rows']} rows)"
            )
            
            if col_data.dtype in ['int64', 'float64']:
                mean_val = col_data.mean()
                median_val = col_data.median()
                reasoning['thought_process'].append(
                    f"Column is numeric: mean={mean_val:.2f}, median={median_val:.2f}"
                )
                
                if abs(mean_val - median_val) < (col_data.std() * 0.1):
                    reasoning['thought_process'].append(
                        "Mean and median are similar → distribution appears normal"
                    )
                else:
                    reasoning['thought_process'].append(
                        "Mean differs from median → distribution may be skewed, prefer median"
                    )
            
            if null_pct < 5:
                reasoning['thought_process'].append(
                    "Low null percentage → filling is safe and preserves data"
                )
            elif null_pct > 30:
                reasoning['thought_process'].append(
                    "High null percentage → consider if column is useful or if deletion is better"
                )
            
            # Alternatives
            if method == 'FILL_MEDIAN':
                reasoning['alternatives'] = ['FILL_MEAN (if distribution is normal)', 'DELETE_ROWS (if nulls are non-random)']
            
            # Risks
            if null_pct > 20:
                reasoning['risks'].append(f"Imputing {null_pct:.1f}% of data may introduce bias")
            if method == 'DELETE_ROWS':
                reasoning['risks'].append(f"Will lose {action['affected_rows']} records")
                
        elif action['type'] == 'REMOVE_DUPLICATES':
            dup_pct = (action['affected_rows'] / len(df)) * 100
            reasoning['thought_process'].append(
                f"Found {action['affected_rows']} duplicate rows ({dup_pct:.1f}% of dataset)"
            )
            reasoning['thought_process'].append(
                "Duplicates reduce data quality and may skew analysis"
            )
            reasoning['alternatives'] = ['KEEP_LAST (if newer records preferred)', 'Manual review (if duplicates seem intentional)']
            reasoning['risks'] = ['May remove legitimate repeated measurements'] if dup_pct > 20 else []
            
        return reasoning
    
    def propose_healing_actions(self, df, quality_report):
        """Propose specific healing actions with confidence scoring and reasoning"""
        
        # Get learning insights for healing decisions
        learning_insights = self.learning_system.get_agent_recommendations(
            "DataHealingAgent", 
            f"healing_{quality_report.get('dataset_name', 'unknown')}"
        )
        
        healing_plan = {
            'actions': [],
            'estimated_impact': {},
            'execution_order': [],
            'learning_applied': True,
            'learned_preferences': learning_insights.get('preferred_methods', {}),
            'learning_confidence': learning_insights.get('overall_confidence', 0.5),
            'auto_executed_actions': [],  # Track auto-executed actions
            'pending_approval_actions': []  # Track actions needing approval
        }
        
        for issue in quality_report['issues']:
            if issue['type'] == 'NULL_VALUES' and issue['severity'] in ['HIGH', 'MEDIUM', 'LOW']:
                action = {
                    'type': 'HANDLE_NULLS',
                    'column': issue['column'],
                    'affected_rows': issue['affected_rows'],
                    'severity': issue['severity'],
                    'options': []
                }
                
                # Determine best null handling strategy
                col_data = df[issue['column']]
                if col_data.dtype in ['int64', 'float64']:
                    # Choose median for numeric (generally safer)
                    recommended_method = 'FILL_MEDIAN'
                    action['options'] = [
                        {'method': 'FILL_MEAN', 'description': f"Fill with mean value ({col_data.mean():.2f})"},
                        {'method': 'FILL_MEDIAN', 'description': f"Fill with median value ({col_data.median():.2f})"},
                        {'method': 'DELETE_ROWS', 'description': f"Delete {issue['affected_rows']} rows with null values"}
                    ]
                else:
                    # Choose mode/default for categorical
                    recommended_method = 'FILL_MODE'
                    action['options'] = [
                        {'method': 'FILL_MODE', 'description': f"Fill with most common value"},
                        {'method': 'FILL_DEFAULT', 'description': f"Fill with default value 'Unknown'"},
                        {'method': 'DELETE_ROWS', 'description': f"Delete {issue['affected_rows']} rows with null values"}
                    ]
                
                # Calculate confidence and reasoning
                action['recommended_method'] = recommended_method
                action['confidence'] = self.calculate_action_confidence(df, action, recommended_method, learning_insights)
                action['reasoning'] = self.generate_action_reasoning(df, action, recommended_method, action['confidence'])
                action['risk_level'] = self._determine_risk_level(action['confidence'], issue['severity'])
                
                healing_plan['actions'].append(action)
            
            elif issue['type'] == 'DUPLICATES' and issue['severity'] in ['HIGH', 'MEDIUM', 'LOW']:
                recommended_method = 'KEEP_FIRST'
                action = {
                    'type': 'REMOVE_DUPLICATES',
                    'affected_rows': issue['affected_rows'],
                    'severity': issue['severity'],
                    'options': [
                        {'method': 'KEEP_FIRST', 'description': 'Keep first occurrence of duplicates'},
                        {'method': 'KEEP_LAST', 'description': 'Keep last occurrence of duplicates'}
                    ],
                    'recommended_method': recommended_method
                }
                
                action['confidence'] = self.calculate_action_confidence(df, action, recommended_method, learning_insights)
                action['reasoning'] = self.generate_action_reasoning(df, action, recommended_method, action['confidence'])
                action['risk_level'] = self._determine_risk_level(action['confidence'], issue['severity'])
                
                healing_plan['actions'].append(action)
            
            elif issue['type'] == 'OUTLIERS' and issue['severity'] in ['HIGH', 'MEDIUM']:
                recommended_method = 'CAP_OUTLIERS'
                action = {
                    'type': 'HANDLE_OUTLIERS',
                    'column': issue['column'],
                    'affected_rows': issue['affected_rows'],
                    'severity': issue['severity'],
                    'options': [
                        {'method': 'CAP_OUTLIERS', 'description': 'Cap outliers to 95th percentile'},
                        {'method': 'REMOVE_OUTLIERS', 'description': 'Remove outlier records'},
                        {'method': 'LOG_TRANSFORM', 'description': 'Apply log transformation'}
                    ],
                    'recommended_method': recommended_method
                }
                
                action['confidence'] = self.calculate_action_confidence(df, action, recommended_method, learning_insights)
                action['reasoning'] = self.generate_action_reasoning(df, action, recommended_method, action['confidence'])
                action['risk_level'] = self._determine_risk_level(action['confidence'], issue['severity'])
                
                healing_plan['actions'].append(action)
        
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
        
        # Categorize actions by confidence for auto-execution
        for action in healing_plan['actions']:
            confidence = action.get('confidence', 0.0)
            risk_level = action.get('risk_level', 'MEDIUM')
            
            # AUTO-EXECUTE: High confidence (>90%) AND Low risk
            if confidence >= 0.90 and risk_level == 'LOW':
                healing_plan['auto_executed_actions'].append(action)
            else:
                # REQUIRE APPROVAL: Everything else
                healing_plan['pending_approval_actions'].append(action)
        
        # Get AI recommendations for execution order (for pending actions)
        if healing_plan['pending_approval_actions']:
            ai_prompt = f"""
            As a data healing specialist, recommend the optimal execution order for these data cleaning actions:
            
            Actions to perform: {json.dumps([a['type'] for a in healing_plan['pending_approval_actions']], indent=2)}
            
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
    
    def _determine_risk_level(self, confidence, severity):
        """Determine risk level based on confidence and severity"""
        # High confidence + Low severity = LOW risk
        # Low confidence or High severity = HIGH risk
        
        if confidence >= 0.90 and severity == 'LOW':
            return 'LOW'
        elif confidence >= 0.75 and severity in ['LOW', 'MEDIUM']:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
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

    def execute_healing_actions(self, df, healing_plan):
        """Execute multiple healing actions on a dataset"""
        current_df = df.copy()
        execution_logs = []
        total_records_affected = 0

        for action in healing_plan.get('actions', []):
            method = action.get('recommended_method', action.get('method'))
            if method:
                healed_df, exec_log = self.execute_healing_action(current_df, action, method)
                current_df = healed_df
                execution_logs.append(exec_log)
                total_records_affected += exec_log.get('records_affected', 0)

        healing_summary = {
            'total_actions_executed': len(execution_logs),
            'total_records_affected': total_records_affected,
            'final_shape': current_df.shape,
            'execution_logs': execution_logs,
            'processing_method': 'standard'
        }

        return current_df, healing_summary

    def execute_healing_actions_chunked(self, df, healing_plan, chunk_size=50000):
        """Execute healing actions in chunks for large datasets"""
        if len(df) <= chunk_size:
            # For smaller datasets, use regular processing
            return self.execute_healing_actions(df, healing_plan)

        self.logger.info(f"Processing large dataset ({len(df)} rows) in chunks of {chunk_size}")

        healed_chunks = []
        total_processed = 0
        total_affected = 0

        # Process in chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunk_start = i
            chunk_end = min(i + chunk_size, len(df))

            self.logger.info(f"Processing chunk {i//chunk_size + 1}: rows {chunk_start}-{chunk_end}")

            # Execute healing actions on this chunk
            healed_chunk, chunk_log = self.execute_healing_actions(chunk, healing_plan)

            healed_chunks.append(healed_chunk)
            total_processed += len(chunk)
            total_affected += chunk_log.get('total_records_affected', 0)

            # Log progress (optional - only if dataset_id available)
            # self.db.log_agent_action(
            #     dataset_id, self.name,
            #     "CHUNK_HEALING_PROGRESS",
            #     f"Processing chunk {i//chunk_size + 1}: rows {chunk_start}-{chunk_end}",
            #     "EXECUTING"
            # )

        # Combine all chunks
        final_df = pd.concat(healed_chunks, ignore_index=True)

        # Final summary log
        healing_summary = {
            'total_chunks': len(healed_chunks),
            'total_processed': total_processed,
            'total_records_affected': total_affected,
            'final_shape': final_df.shape,
            'processing_method': 'chunked'
        }

        # Final summary log (optional)
        # self.db.log_agent_action(
        #     dataset_id, self.name,
        #     "CHUNKED_HEALING_COMPLETE",
        #     f"Processed {len(healed_chunks)} chunks, {total_processed} total records, {total_affected} affected",
        #     "COMPLETED"
        # )

        return final_df, healing_summary
    
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
            
            # ========== AUTO-EXECUTE HIGH-CONFIDENCE ACTIONS ==========
            auto_executed_results = []
            current_df = df.copy()
            
            if healing_plan.get('auto_executed_actions'):
                auto_exec_log_id = self.db.log_agent_action(
                    dataset_id, self.healing_agent.name,
                    "AUTO_EXECUTION_START",
                    f"Auto-executing {len(healing_plan['auto_executed_actions'])} high-confidence actions",
                    "EXECUTING"
                )
                
                for action in healing_plan['auto_executed_actions']:
                    method = action.get('recommended_method')
                    confidence = action.get('confidence', 0.0)
                    
                    # Execute the action
                    current_df, exec_log = self.healing_agent.execute_healing_action(current_df, action, method)
                    
                    # Log auto-execution
                    self.db.log_agent_action(
                        dataset_id, self.healing_agent.name,
                        f"AUTO_EXECUTED_{action['type']}",
                        f"Auto-executed {action['type']} (confidence: {confidence:.0%}) for column '{action.get('column', 'multiple')}' - {exec_log['message']}",
                        "COMPLETED" if exec_log['success'] else "FAILED",
                        human_approval="NOT_REQUIRED"
                    )
                    
                    auto_executed_results.append({
                        'action': action,
                        'execution_log': exec_log,
                        'confidence': confidence
                    })
                
                self.db.update_agent_action(auto_exec_log_id, "COMPLETED")
                
                # Update quality score after auto-execution
                if auto_executed_results:
                    new_quality_score = self.healing_agent.recalculate_quality_score(current_df)
                    quality_improvement = new_quality_score - quality_report['quality_score']
                    
                    self.db.log_agent_action(
                        dataset_id, self.healing_agent.name,
                        "AUTO_EXECUTION_COMPLETE",
                        f"Auto-executed {len(auto_executed_results)} actions. Quality improved: {quality_report['quality_score']:.1f}% → {new_quality_score:.1f}% (+{quality_improvement:.1f}%)",
                        "COMPLETED"
                    )
            
            # Store auto-execution results in processing log
            processing_log['auto_executed_actions'] = auto_executed_results
            
            # ========== PREPARE ACTIONS REQUIRING APPROVAL ==========
            # Log each pending approval action
            for i, action in enumerate(healing_plan.get('pending_approval_actions', [])):
                action_id = self.db.log_agent_action(
                    dataset_id, self.healing_agent.name,
                    f"HEALING_PROPOSED_{action['type']}",
                    f"Proposed {action['type']} (confidence: {action.get('confidence', 0):.0%}) for column '{action.get('column', 'multiple')}' affecting {action.get('affected_rows', 0)} records",
                    "PENDING_APPROVAL"
                )
                
                # Determine which actions need human approval based on severity and impact
                affected_percentage = (action.get('affected_rows', 0) / len(df) * 100) if len(df) > 0 else 0
                
                # Use severity from action if available, otherwise default to LOW
                action_severity = action.get('severity', 'LOW')
                
                # Determine if approval is needed based on severity thresholds
                needs_approval = False
                
                if action_severity == 'HIGH':
                    # Always require approval for HIGH severity
                    needs_approval = True
                elif action_severity == 'MEDIUM':
                    # Require approval for MEDIUM severity (regardless of percentage)
                    needs_approval = True
                elif action_severity == 'LOW' and affected_percentage > 10.0:
                    # Only require approval for LOW severity if affecting >10%
                    needs_approval = True
                
                if needs_approval:
                    processing_log['human_approvals_needed'].append({
                        'action_id': action_id,
                        'action_type': action['type'],
                        'column': action.get('column', 'N/A'),
                        'impact': f"Affects {action.get('affected_rows', 0)} records ({affected_percentage:.1f}%)",
                        'severity': action_severity,
                        'recommendation': action['options'][0] if action.get('options') else None,
                        'options': action.get('options', [])
                    })
                else:
                    # Auto-approve low impact actions
                    self.db.update_agent_action(action_id, "AUTO_APPROVED", "AUTO_APPROVED")
                    self.db.log_agent_action(
                        dataset_id, self.name,
                        f"AUTO_APPROVED_{action['type']}",
                        f"Auto-approved {action_severity} severity action affecting {action.get('affected_rows', 0)} records ({affected_percentage:.1f}%) - below approval threshold",
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
            As a Pipeline Orchestrator, provide a concise executive summary (under 50 words):
            
            Dataset: {dataset_name}
            Quality Score: {quality_report['quality_score']:.1f}/100
            Issues Found: {len(quality_report['issues'])}
            Actions Requiring Approval: {len(processing_log['human_approvals_needed'])}
            
            Provide a single paragraph summary highlighting key findings and next steps.
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
        processing_log['current_df'] = current_df if 'current_df' in locals() else df  # Return df with auto-executed changes
        
        return processing_log, dataset_id
