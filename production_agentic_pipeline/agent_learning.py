"""
Agent Learning System - Enables AI agents to learn from their actions and improve over time
"""
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from database import ProductionDatabase

class AgentLearningSystem:
    """
    Core learning engine that enables agents to learn from their actions,
    successes, failures, and human feedback to continuously improve performance.
    """
    
    def __init__(self, db: ProductionDatabase):
        self.db = db
        self.learning_data = {}
        self.initialize_learning_tables()
        self.load_learning_history()
        print("AgentLearningSystem initialized - agents can now learn and adapt!")
    
    def initialize_learning_tables(self):
        """Create database tables for agent learning if they don't exist"""
        try:
            # Agent learning actions table
            learning_table_query = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AgentLearning' AND xtype='U')
            CREATE TABLE AgentLearning (
                ID int IDENTITY(1,1) PRIMARY KEY,
                AgentName nvarchar(100) NOT NULL,
                ActionType nvarchar(100) NOT NULL,
                Context nvarchar(max),
                InputData nvarchar(max),
                OutputData nvarchar(max),
                Outcome nvarchar(50) NOT NULL,
                SuccessScore float DEFAULT 0.0,
                ExecutionTime float DEFAULT 0.0,
                ErrorMessage nvarchar(max),
                LearningTimestamp datetime DEFAULT GETDATE(),
                INDEX IX_AgentLearning_Agent (AgentName),
                INDEX IX_AgentLearning_Action (ActionType),
                INDEX IX_AgentLearning_Outcome (Outcome)
            )
            """
            
            # Human feedback table
            feedback_table_query = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='HumanFeedback' AND xtype='U')
            CREATE TABLE HumanFeedback (
                ID int IDENTITY(1,1) PRIMARY KEY,
                AgentName nvarchar(100) NOT NULL,
                ActionID int,
                FeedbackType nvarchar(50),
                FeedbackText nvarchar(max),
                Rating int CHECK (Rating >= 1 AND Rating <= 5),
                Context nvarchar(max),
                FeedbackTimestamp datetime DEFAULT GETDATE(),
                INDEX IX_HumanFeedback_Agent (AgentName),
                INDEX IX_HumanFeedback_Rating (Rating)
            )
            """
            
            # Learning patterns table
            patterns_table_query = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='LearningPatterns' AND xtype='U')
            CREATE TABLE LearningPatterns (
                ID int IDENTITY(1,1) PRIMARY KEY,
                AgentName nvarchar(100) NOT NULL,
                PatternType nvarchar(100) NOT NULL,
                PatternData nvarchar(max) NOT NULL,
                SuccessRate float DEFAULT 0.0,
                Confidence float DEFAULT 0.0,
                LastUpdated datetime DEFAULT GETDATE(),
                INDEX IX_LearningPatterns_Agent (AgentName),
                INDEX IX_LearningPatterns_Type (PatternType)
            )
            """
            
            self.db.execute_query(learning_table_query)
            self.db.execute_query(feedback_table_query)
            self.db.execute_query(patterns_table_query)
            
            print("✅ Learning database tables initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize learning tables: {str(e)}")
    
    def log_agent_action(self, agent_name: str, action_type: str, context: str, 
                        input_data: Dict, output_data: Dict, success: bool, 
                        execution_time: float = 0.0, error_message: str = None):
        """Log an agent action for learning analysis"""
        try:
            success_score = 1.0 if success else 0.0
            outcome = "SUCCESS" if success else "FAILED"
            
            query = """
                INSERT INTO AgentLearning 
                (AgentName, ActionType, Context, InputData, OutputData, Outcome, 
                 SuccessScore, ExecutionTime, ErrorMessage, LearningTimestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db.execute_query(query, (
                agent_name, action_type, context,
                json.dumps(input_data), json.dumps(output_data),
                outcome, success_score, execution_time,
                error_message, datetime.now().isoformat()
            ))
            
            # Update learning patterns
            self.update_learning_patterns(agent_name, action_type, success, context)
            
        except Exception as e:
            print(f"Error logging agent action: {str(e)}")
    
    def update_learning_patterns(self, agent_name: str, action_type: str, success: bool, context: str):
        """Update learning patterns based on action outcomes"""
        try:
            # Get existing pattern or create new one
            pattern_key = f"{action_type}_{context}"
            
            query = """
                SELECT PatternData, SuccessRate FROM LearningPatterns 
                WHERE AgentName = ? AND PatternType = ?
            """
            
            result = self.db.execute_query(query, (agent_name, pattern_key))
            
            if result and len(result) > 0:
                # Update existing pattern
                pattern_data = json.loads(result[0][0])
                current_success_rate = result[0][1]
                
                pattern_data['total_attempts'] = pattern_data.get('total_attempts', 0) + 1
                if success:
                    pattern_data['successful_attempts'] = pattern_data.get('successful_attempts', 0) + 1
                
                new_success_rate = pattern_data['successful_attempts'] / pattern_data['total_attempts']
                confidence = min(pattern_data['total_attempts'] / 10.0, 1.0)  # Max confidence after 10 attempts
                
                update_query = """
                    UPDATE LearningPatterns 
                    SET PatternData = ?, SuccessRate = ?, Confidence = ?, LastUpdated = ?
                    WHERE AgentName = ? AND PatternType = ?
                """
                
                self.db.execute_query(update_query, (
                    json.dumps(pattern_data), new_success_rate, confidence,
                    datetime.now().isoformat(), agent_name, pattern_key
                ))
                
            else:
                # Create new pattern
                pattern_data = {
                    'total_attempts': 1,
                    'successful_attempts': 1 if success else 0,
                    'context': context,
                    'action_type': action_type
                }
                
                success_rate = 1.0 if success else 0.0
                confidence = 0.1  # Low confidence with single data point
                
                insert_query = """
                    INSERT INTO LearningPatterns 
                    (AgentName, PatternType, PatternData, SuccessRate, Confidence, LastUpdated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                
                self.db.execute_query(insert_query, (
                    agent_name, pattern_key, json.dumps(pattern_data),
                    success_rate, confidence, datetime.now().isoformat()
                ))
                
        except Exception as e:
            print(f"Error updating learning patterns: {str(e)}")
    
    def get_agent_recommendations(self, agent_name: str, context: str = None) -> Dict[str, Any]:
        """Get learned recommendations for an agent based on historical performance"""
        try:
            # Get learning patterns for this agent
            query = """
                SELECT PatternType, PatternData, SuccessRate, Confidence 
                FROM LearningPatterns 
                WHERE AgentName = ? AND Confidence > 0.3
                ORDER BY SuccessRate DESC, Confidence DESC
            """
            
            results = self.db.execute_query(query, (agent_name,))
            
            recommendations = {
                'agent_name': agent_name,
                'context': context,
                'confidence_level': 'learning',
                'recommendations': {},
                'learned_preferences': {}
            }
            
            if results and len(results) > 0:
                total_confidence = 0
                successful_patterns = []
                
                for pattern_type, pattern_data, success_rate, confidence in results:
                    data = json.loads(pattern_data)
                    total_confidence += confidence
                    
                    if success_rate > 0.7:  # High success rate
                        successful_patterns.append({
                            'pattern': pattern_type,
                            'success_rate': success_rate,
                            'confidence': confidence,
                            'data': data
                        })
                
                # Generate specific recommendations based on agent type
                if agent_name == 'DataQualityAgent':
                    recommendations.update(self._generate_quality_recommendations(successful_patterns))
                elif agent_name == 'DataHealingAgent':
                    recommendations.update(self._generate_healing_recommendations(successful_patterns))
                elif agent_name == 'PipelineOrchestrator':
                    recommendations.update(self._generate_orchestrator_recommendations(successful_patterns))
                
                recommendations['overall_confidence'] = min(total_confidence / len(results), 1.0)
                recommendations['learned_patterns_count'] = len(successful_patterns)
            
            # Get human feedback insights
            feedback_insights = self.analyze_human_feedback(agent_name)
            if feedback_insights:
                recommendations['human_feedback_insights'] = feedback_insights
            
            # Get rejection pattern insights
            rejection_insights = self.analyze_rejection_patterns(agent_name)
            if rejection_insights:
                recommendations['rejection_pattern_insights'] = rejection_insights
                
                # Adjust recommendations based on rejection patterns
                if agent_name == 'DataHealingAgent' and rejection_insights.get('preferred_alternatives'):
                    # Boost confidence in human-preferred alternatives
                    for alt_method in rejection_insights['preferred_alternatives']:
                        recommendations['human_preferred_alternatives'] = rejection_insights['preferred_alternatives']
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting agent recommendations: {str(e)}")
            return {'agent_name': agent_name, 'context': context, 'error': str(e)}
    
    def _generate_quality_recommendations(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Generate quality agent specific recommendations"""
        recommendations = {
            'severity_thresholds': {'null_high': 15, 'null_medium': 8, 'duplicate_high': 10, 'duplicate_medium': 5},
            'focus_areas': ['completeness', 'consistency', 'validity'],
            'quality_weights': {'completeness': 0.4, 'consistency': 0.3, 'uniqueness': 0.3}
        }
        
        # Analyze successful patterns
        for pattern in patterns:
            if 'null_handling' in pattern['pattern'] and pattern['success_rate'] > 0.8:
                # Learned that higher thresholds work better
                recommendations['severity_thresholds']['null_high'] = min(20, recommendations['severity_thresholds']['null_high'] + 2)
            
            if 'quality_analysis' in pattern['pattern']:
                if pattern['data'].get('context') == 'financial_data':
                    recommendations['focus_areas'].append('accuracy')
        
        return recommendations
    
    def _generate_healing_recommendations(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Generate healing agent specific recommendations"""
        recommendations = {
            'preferred_methods': {
                'null_handling': {'numeric': 'FILL_MEDIAN', 'categorical': 'FILL_MODE'},
                'duplicate_handling': 'KEEP_FIRST',
                'outlier_handling': 'CAP_OUTLIERS'
            },
            'execution_order': ['HANDLE_NULLS', 'REMOVE_DUPLICATES', 'HANDLE_OUTLIERS']
        }
        
        # Learn from successful healing patterns
        for pattern in patterns:
            if 'FILL_MEDIAN' in pattern['pattern'] and pattern['success_rate'] > 0.9:
                recommendations['preferred_methods']['null_handling']['numeric'] = 'FILL_MEDIAN'
            elif 'FILL_MEAN' in pattern['pattern'] and pattern['success_rate'] > 0.85:
                recommendations['preferred_methods']['null_handling']['numeric'] = 'FILL_MEAN'
        
        return recommendations
    
    def _generate_orchestrator_recommendations(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Generate orchestrator specific recommendations"""
        recommendations = {
            'auto_approval_threshold': 0.1,  # 10% of data affected
            'quality_score_threshold': 85.0,
            'safe_actions': ['HANDLE_NULLS', 'REMOVE_DUPLICATES'],
            'human_approval_needed': ['HANDLE_OUTLIERS', 'DELETE_COLUMNS']
        }
        
        # Learn safe auto-approval patterns
        for pattern in patterns:
            if 'auto_approved' in pattern['pattern'] and pattern['success_rate'] > 0.95:
                # Can be more aggressive with auto-approval
                recommendations['auto_approval_threshold'] = min(0.15, recommendations['auto_approval_threshold'] + 0.02)
        
        return recommendations
    
    def store_human_feedback(self, agent_name: str, action_id: int = None, 
                           feedback_type: str = 'general', feedback_text: str = '',
                           rating: int = 3, context: str = ''):
        """Store human feedback for agent learning"""
        try:
            query = """
                INSERT INTO HumanFeedback 
                (AgentName, ActionID, FeedbackType, FeedbackText, Rating, Context, FeedbackTimestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db.execute_query(query, (
                agent_name, action_id, feedback_type, feedback_text,
                rating, context, datetime.now().isoformat()
            ))
            
            print(f"✅ Human feedback stored for {agent_name}")
            
            # Update learning patterns based on feedback
            self.update_patterns_from_feedback(agent_name, rating, feedback_text, context)
            
        except Exception as e:
            print(f"Error storing human feedback: {str(e)}")

    def parse_and_store_nlp_feedback(self, agent_name: str, feedback_text: str, rating: int = 3, context: str = '', action_id: int = None):
        """Parse free-text human feedback into structured signals using light NLP heuristics
        and store both raw and structured feedback for the learning system.
        This keeps everything local and deterministic (no external NLP required).
        """
        try:
            raw_text = feedback_text or ''
            text = raw_text.lower()

            parsed = {
                'suggested_method': None,
                'target': None,
                'severity_hint': None,
                'notes': raw_text
            }

            # simple heuristics
            if 'median' in text:
                parsed['suggested_method'] = 'FILL_MEDIAN'
            elif 'mean' in text or 'average' in text:
                parsed['suggested_method'] = 'FILL_MEAN'
            elif 'mode' in text or 'most common' in text:
                parsed['suggested_method'] = 'FILL_MODE'

            if 'duplicate' in text:
                parsed['target'] = 'duplicates'
            if 'null' in text or 'missing' in text or 'nan' in text:
                parsed['target'] = 'nulls'
            if 'outlier' in text or 'cap' in text or 'remove outlier' in text:
                parsed['target'] = 'outliers'

            if 'high' in text or 'severe' in text:
                parsed['severity_hint'] = 'HIGH'
            elif 'low' in text or 'minor' in text:
                parsed['severity_hint'] = 'LOW'

            # store raw feedback in HumanFeedback and also log parsed insight into LearningPatterns
            # store raw
            self.store_human_feedback(agent_name=agent_name, action_id=action_id, feedback_type='nlp_parsed', feedback_text=raw_text, rating=rating, context=context)

            # apply parsed signals to update preferences/patterns
            try:
                if parsed['suggested_method']:
                    # weakly reinforce pattern for agent to prefer suggested method
                    self.update_method_preference(agent_name, parsed['suggested_method'], True)

                # Reinforcement depending on rating
                if rating >= 4:
                    self.reinforce_patterns(agent_name, context or parsed.get('target',''), 1.05)
                elif rating <= 2:
                    self.reinforce_patterns(agent_name, context or parsed.get('target',''), 0.9)
            except Exception as e:
                print(f"Error applying parsed feedback signals: {str(e)}")

            return parsed

        except Exception as e:
            print(f"Error parsing/storing NLP feedback: {str(e)}")
            return None
    
    def update_patterns_from_feedback(self, agent_name: str, rating: int, feedback_text: str, context: str):
        """Update learning patterns based on human feedback"""
        try:
            # Positive feedback (rating 4-5) reinforces current patterns
            # Negative feedback (rating 1-2) suggests pattern changes needed
            
            if rating >= 4:
                # Reinforce successful patterns
                self.reinforce_patterns(agent_name, context, 1.1)  # 10% boost
            elif rating <= 2:
                # Penalize unsuccessful patterns
                self.reinforce_patterns(agent_name, context, 0.9)  # 10% reduction
            
            # Analyze feedback text for specific improvements
            if 'median' in feedback_text.lower():
                self.update_method_preference(agent_name, 'median_preferred', True)
            elif 'mean' in feedback_text.lower() and rating <= 2:
                self.update_method_preference(agent_name, 'mean_preferred', False)
                
        except Exception as e:
            print(f"Error updating patterns from feedback: {str(e)}")
    
    def reinforce_patterns(self, agent_name: str, context: str, multiplier: float):
        """Reinforce or penalize patterns based on feedback"""
        try:
            query = """
                UPDATE LearningPatterns 
                SET SuccessRate = SuccessRate * ?, 
                    Confidence = CASE 
                        WHEN Confidence * ? > 1.0 THEN 1.0 
                        ELSE Confidence * ? 
                    END,
                    LastUpdated = ?
                WHERE AgentName = ? AND PatternType LIKE ?
            """
            
            self.db.execute_query(query, (
                multiplier, multiplier, multiplier,
                datetime.now().isoformat(), agent_name, f'%{context}%'
            ))
            
        except Exception as e:
            print(f"Error reinforcing patterns: {str(e)}")
    
    def update_method_preference(self, agent_name: str, preference: str, value: bool):
        """Update method preferences based on feedback"""
        # This would update specific method preferences
        # Implementation depends on how preferences are stored
        pass
    
    def analyze_human_feedback(self, agent_name: str) -> Dict[str, Any]:
        """Analyze human feedback patterns for an agent"""
        try:
            query = """
                SELECT FeedbackType, AVG(CAST(Rating as float)) as AvgRating, COUNT(*) as FeedbackCount,
                       STRING_AGG(FeedbackText, '; ') as AllFeedback
                FROM HumanFeedback 
                WHERE AgentName = ? AND FeedbackTimestamp >= ?
                GROUP BY FeedbackType
            """
            
            cutoff_date = datetime.now() - timedelta(days=30)
            results = self.db.execute_query(query, (agent_name, cutoff_date.isoformat()))
            
            analysis = {
                'overall_satisfaction': 0.0,
                'feedback_count': 0,
                'common_suggestions': [],
                'improvement_areas': []
            }
            
            if results and len(results) > 0:
                total_rating = 0
                total_count = 0
                
                for feedback_type, avg_rating, count, all_feedback in results:
                    total_rating += avg_rating * count
                    total_count += count
                    
                    if avg_rating < 3.0:
                        analysis['improvement_areas'].append(feedback_type)
                    
                    # Simple keyword extraction from feedback
                    if all_feedback:
                        words = all_feedback.lower().split()
                        common_words = ['median', 'mean', 'conservative', 'aggressive', 'careful']
                        for word in common_words:
                            if word in words:
                                analysis['common_suggestions'].append(word)
                
                if total_count > 0:
                    analysis['overall_satisfaction'] = total_rating / total_count
                    analysis['feedback_count'] = total_count
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing human feedback: {str(e)}")
            return {}
            
    def analyze_rejection_patterns(self, agent_name: str) -> Dict[str, Any]:
        """Analyze patterns in human rejections to improve future recommendations"""
        try:
            # Get rejection data from the last 60 days
            query = """
                SELECT ActionType, ActionDetails, FeedbackText, Rating, Context
                FROM AgentActions aa
                LEFT JOIN HumanFeedback hf ON aa.ID = hf.ActionID
                WHERE aa.AgentName = 'HumanOperator' 
                AND aa.ActionType LIKE '%REJECTED%'
                AND aa.ActionTimestamp >= ?
                ORDER BY aa.ActionTimestamp DESC
            """
            
            cutoff_date = datetime.now() - timedelta(days=60)
            results = self.db.execute_query(query, (cutoff_date.isoformat(),))
            
            rejection_analysis = {
                'total_rejections': 0,
                'rejection_patterns': {},
                'preferred_alternatives': {},
                'common_rejection_reasons': [],
                'learning_insights': []
            }
            
            if results and len(results) > 0:
                rejection_analysis['total_rejections'] = len(results)
                
                for action_type, action_details, feedback_text, rating, context in results:
                    # Extract rejected action type
                    if 'REJECTED_' in action_type:
                        rejected_action = action_type.replace('REJECTED_', '').replace('BULK_REJECTED_', '').replace('_WITH_ALTERNATIVE', '')
                        
                        if rejected_action not in rejection_analysis['rejection_patterns']:
                            rejection_analysis['rejection_patterns'][rejected_action] = 0
                        rejection_analysis['rejection_patterns'][rejected_action] += 1
                    
                    # Extract preferred alternatives from feedback
                    if feedback_text:
                        feedback_lower = feedback_text.lower()
                        
                        # Look for alternative method mentions
                        alternative_methods = [
                            'FILL_MEAN', 'FILL_MEDIAN', 'FILL_MODE', 'FILL_FORWARD', 'FILL_BACKWARD',
                            'DROP_ROWS', 'DROP_COLUMNS', 'OUTLIER_IQR', 'OUTLIER_ZSCORE', 
                            'DUPLICATE_REMOVE', 'DUPLICATE_KEEP_FIRST'
                        ]
                        
                        for method in alternative_methods:
                            if method.lower() in feedback_lower:
                                if method not in rejection_analysis['preferred_alternatives']:
                                    rejection_analysis['preferred_alternatives'][method] = 0
                                rejection_analysis['preferred_alternatives'][method] += 1
                    
                    # Extract common rejection reasons
                    if feedback_text:
                        # Simple keyword extraction for rejection reasons
                        reason_keywords = {
                            'too aggressive': 'too_aggressive',
                            'too conservative': 'too_conservative', 
                            'wrong method': 'wrong_method',
                            'data loss': 'data_loss_concern',
                            'accuracy': 'accuracy_concern',
                            'manual review': 'needs_manual_review'
                        }
                        
                        feedback_lower = feedback_text.lower()
                        for keyword, reason in reason_keywords.items():
                            if keyword in feedback_lower:
                                rejection_analysis['common_rejection_reasons'].append(reason)
                
                # Generate learning insights
                if rejection_analysis['rejection_patterns']:
                    most_rejected = max(rejection_analysis['rejection_patterns'], 
                                      key=rejection_analysis['rejection_patterns'].get)
                    rejection_analysis['learning_insights'].append(
                        f"Most frequently rejected action: {most_rejected} "
                        f"({rejection_analysis['rejection_patterns'][most_rejected]} times)"
                    )
                
                if rejection_analysis['preferred_alternatives']:
                    most_preferred = max(rejection_analysis['preferred_alternatives'], 
                                       key=rejection_analysis['preferred_alternatives'].get)
                    rejection_analysis['learning_insights'].append(
                        f"Most preferred alternative: {most_preferred} "
                        f"({rejection_analysis['preferred_alternatives'][most_preferred]} times)"
                    )
            
            return rejection_analysis
            
        except Exception as e:
            print(f"Error analyzing rejection patterns: {str(e)}")
            return {}
            
        except Exception as e:
            print(f"Error analyzing rejection patterns: {str(e)}")
            return {}
    
    def load_learning_history(self):
        """Load existing learning data from database"""
        try:
            query = """
                SELECT AgentName, ActionType, AVG(SuccessScore) as AvgSuccess, COUNT(*) as ActionCount
                FROM AgentLearning 
                WHERE LearningTimestamp >= ?
                GROUP BY AgentName, ActionType
            """
            
            cutoff_date = datetime.now() - timedelta(days=90)  # Last 90 days
            results = self.db.execute_query(query, (cutoff_date.isoformat(),))
            
            if results:
                for agent_name, action_type, avg_success, action_count in results:
                    if agent_name not in self.learning_data:
                        self.learning_data[agent_name] = {}
                    
                    self.learning_data[agent_name][action_type] = {
                        'success_rate': float(avg_success),
                        'action_count': int(action_count),
                        'confidence': min(action_count / 10.0, 1.0)
                    }
            
            print(f"📊 Loaded learning history for {len(self.learning_data)} agents")
            
        except Exception as e:
            print(f"Error loading learning history: {str(e)}")
    
    def get_success_rate(self, agent_name: str, days_back: int = 30) -> float:
        """Get success rate for an agent over specified period"""
        try:
            query = """
                SELECT AVG(SuccessScore) as AvgSuccess
                FROM AgentLearning 
                WHERE AgentName = ? AND LearningTimestamp >= ?
            """
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            result = self.db.execute_query(query, (agent_name, cutoff_date.isoformat()))
            
            if result and len(result) > 0 and result[0][0] is not None:
                return float(result[0][0]) * 100.0
            else:
                return 100.0  # Default optimistic value
                
        except Exception as e:
            print(f"Error getting success rate: {str(e)}")
            return 100.0
    
    def get_total_actions_count(self, agent_name: str, days_back: int = 30) -> int:
        """Get total actions count for an agent"""
        try:
            query = """
                SELECT COUNT(*) 
                FROM AgentLearning 
                WHERE AgentName = ? AND LearningTimestamp >= ?
            """
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            result = self.db.execute_query(query, (agent_name, cutoff_date.isoformat()))
            
            if result and len(result) > 0:
                return int(result[0][0])
            else:
                return 0
                
        except Exception as e:
            print(f"Error getting actions count: {str(e)}")
            return 0
    
    def analyze_success_failure_patterns(self, agent_name: str) -> Dict[str, List[str]]:
        """Analyze success and failure patterns for an agent"""
        try:
            query = """
                SELECT ActionType, AVG(SuccessScore) as AvgSuccess, COUNT(*) as Count,
                       STRING_AGG(Context, '; ') as Contexts
                FROM AgentLearning 
                WHERE AgentName = ? AND LearningTimestamp >= ?
                GROUP BY ActionType
                HAVING COUNT(*) >= 3
                ORDER BY AvgSuccess DESC
            """
            
            cutoff_date = datetime.now() - timedelta(days=60)
            results = self.db.execute_query(query, (agent_name, cutoff_date.isoformat()))
            
            patterns = {
                'successful_patterns': [],
                'failure_patterns': []
            }
            
            if results:
                for action_type, avg_success, count, contexts in results:
                    pattern_desc = f"{action_type} (Success: {avg_success*100:.1f}%, Count: {count})"
                    
                    if avg_success >= 0.8:
                        patterns['successful_patterns'].append(pattern_desc)
                    elif avg_success <= 0.4:
                        patterns['failure_patterns'].append(pattern_desc)
            
            return patterns
            
        except Exception as e:
            print(f"Error analyzing patterns: {str(e)}")
            return {'successful_patterns': [], 'failure_patterns': []}
    
    def generate_learning_summary(self, agent_name: str) -> Dict[str, Any]:
        """Generate comprehensive learning summary for an agent"""
        try:
            summary = {
                'agent_name': agent_name,
                'total_actions': self.get_total_actions_count(agent_name, 90),
                'success_rate': self.get_success_rate(agent_name, 90),
                'learning_confidence': 'low',
                'recent_improvements': [],
                'recommended_focus_areas': []
            }
            
            # Determine learning confidence
            if summary['total_actions'] > 50:
                summary['learning_confidence'] = 'high'
            elif summary['total_actions'] > 20:
                summary['learning_confidence'] = 'medium'
            
            # Get patterns
            patterns = self.analyze_success_failure_patterns(agent_name)
            summary['successful_patterns'] = patterns['successful_patterns']
            summary['failure_patterns'] = patterns['failure_patterns']
            
            # Human feedback analysis
            feedback_analysis = self.analyze_human_feedback(agent_name)
            if feedback_analysis:
                summary['human_satisfaction'] = feedback_analysis.get('overall_satisfaction', 3.0)
                summary['feedback_count'] = feedback_analysis.get('feedback_count', 0)
                summary['improvement_areas'] = feedback_analysis.get('improvement_areas', [])
            
            return summary
            
        except Exception as e:
            print(f"Error generating learning summary: {str(e)}")
            return {'agent_name': agent_name, 'error': str(e)}
