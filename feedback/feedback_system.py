"""
Feedback system implementation for SecureInsight.

Provides local feedback collection, storage, and anonymized export capabilities
for continuous system improvement and performance tracking.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
from loguru import logger


@dataclass
class FeedbackData:
    """Data structure for user feedback."""
    feedback_id: str
    query: str
    response: str
    rating: int  # 1-5 scale
    comments: Optional[str]
    timestamp: datetime
    query_metadata: Dict[str, Any]
    processing_time: float
    similarity_scores: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback data to dictionary format."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackData':
        """Create FeedbackData from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class FeedbackSystem:
    """
    Local feedback collection and storage system.
    
    Provides capabilities for:
    - Rating collection (1-5 scale) with optional comments
    - Query metadata and performance metrics logging
    - Anonymized feedback export functionality
    - Local storage without external dependencies
    """
    
    def __init__(self, feedback_dir: str = "feedback/feedback_logs"):
        """
        Initialize feedback system.
        
        Args:
            feedback_dir: Directory to store feedback logs
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.feedback_dir / "raw").mkdir(exist_ok=True)
        (self.feedback_dir / "aggregated").mkdir(exist_ok=True)
        (self.feedback_dir / "exports").mkdir(exist_ok=True)
        
        self.feedback_file = self.feedback_dir / "raw" / "feedback.jsonl"
        
        logger.info(f"Feedback system initialized with directory: {self.feedback_dir}")
    
    def collect_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        comments: Optional[str] = None,
        query_metadata: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0,
        similarity_scores: Optional[List[float]] = None
    ) -> str:
        """
        Collect user feedback and store locally.
        
        Args:
            query: Original user query
            response: System response
            rating: User rating (1-5 scale)
            comments: Optional user comments
            query_metadata: Metadata about the query (e.g., query type, modality)
            processing_time: Time taken to process the query
            similarity_scores: Similarity scores from retrieval
            
        Returns:
            feedback_id: Unique identifier for the feedback entry
            
        Raises:
            ValueError: If rating is not in valid range (1-5)
        """
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5, got {rating}")
        
        feedback_id = str(uuid.uuid4())
        
        feedback_data = FeedbackData(
            feedback_id=feedback_id,
            query=query,
            response=response,
            rating=rating,
            comments=comments,
            timestamp=datetime.now(),
            query_metadata=query_metadata or {},
            processing_time=processing_time,
            similarity_scores=similarity_scores or []
        )
        
        try:
            self._log_feedback_locally(feedback_data)
            logger.info(f"Feedback collected successfully: {feedback_id}")
            return feedback_id
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            raise
    
    def _log_feedback_locally(self, feedback_data: FeedbackData) -> None:
        """
        Log feedback data to local storage.
        
        Args:
            feedback_data: Feedback data to log
        """
        try:
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                json.dump(feedback_data.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log feedback locally: {e}")
            raise
    
    def get_feedback_by_id(self, feedback_id: str) -> Optional[FeedbackData]:
        """
        Retrieve feedback by ID.
        
        Args:
            feedback_id: Unique feedback identifier
            
        Returns:
            FeedbackData if found, None otherwise
        """
        try:
            for feedback in self._read_all_feedback():
                if feedback.feedback_id == feedback_id:
                    return feedback
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve feedback {feedback_id}: {e}")
            return None
    
    def get_feedback_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated feedback metrics.
        
        Returns:
            Dictionary containing feedback metrics
        """
        try:
            feedback_list = list(self._read_all_feedback())
            
            if not feedback_list:
                return {
                    'total_feedback': 0,
                    'average_rating': 0.0,
                    'rating_distribution': {},
                    'average_processing_time': 0.0,
                    'feedback_trends': {}
                }
            
            ratings = [f.rating for f in feedback_list]
            processing_times = [f.processing_time for f in feedback_list]
            
            # Calculate rating distribution
            rating_distribution = {}
            for i in range(1, 6):
                rating_distribution[str(i)] = ratings.count(i)
            
            # Calculate trends by day
            feedback_by_date = {}
            for feedback in feedback_list:
                date_key = feedback.timestamp.date().isoformat()
                if date_key not in feedback_by_date:
                    feedback_by_date[date_key] = []
                feedback_by_date[date_key].append(feedback.rating)
            
            feedback_trends = {}
            for date, daily_ratings in feedback_by_date.items():
                feedback_trends[date] = {
                    'count': len(daily_ratings),
                    'average_rating': sum(daily_ratings) / len(daily_ratings)
                }
            
            metrics = {
                'total_feedback': len(feedback_list),
                'average_rating': sum(ratings) / len(ratings),
                'rating_distribution': rating_distribution,
                'average_processing_time': sum(processing_times) / len(processing_times),
                'feedback_trends': feedback_trends,
                'latest_feedback_date': max(f.timestamp for f in feedback_list).isoformat(),
                'common_themes': self._extract_common_themes(feedback_list)
            }
            
            logger.info(f"Generated feedback metrics for {len(feedback_list)} entries")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to generate feedback metrics: {e}")
            return {}
    
    def _extract_common_themes(self, feedback_list: List[FeedbackData]) -> Dict[str, int]:
        """
        Extract common themes from feedback comments.
        
        Args:
            feedback_list: List of feedback data
            
        Returns:
            Dictionary of themes and their frequencies
        """
        themes = {}
        
        # Simple keyword-based theme extraction
        keywords = {
            'accuracy': ['accurate', 'correct', 'right', 'wrong', 'incorrect'],
            'speed': ['fast', 'slow', 'quick', 'time', 'performance'],
            'relevance': ['relevant', 'irrelevant', 'useful', 'useless', 'helpful'],
            'completeness': ['complete', 'incomplete', 'missing', 'partial'],
            'clarity': ['clear', 'unclear', 'confusing', 'understandable']
        }
        
        for feedback in feedback_list:
            if feedback.comments:
                comment_lower = feedback.comments.lower()
                for theme, theme_keywords in keywords.items():
                    if any(keyword in comment_lower for keyword in theme_keywords):
                        themes[theme] = themes.get(theme, 0) + 1
        
        return themes
    
    def export_anonymized_feedback(self, export_path: Optional[Path] = None) -> Path:
        """
        Export anonymized feedback data.
        
        Args:
            export_path: Optional custom export path
            
        Returns:
            Path to exported file
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.feedback_dir / "exports" / f"anonymized_feedback_{timestamp}.json"
        
        try:
            feedback_list = list(self._read_all_feedback())
            anonymized_data = []
            
            for feedback in feedback_list:
                # Anonymize sensitive data
                anonymized = {
                    'feedback_id': self._hash_string(feedback.feedback_id),
                    'query_hash': self._hash_string(feedback.query),
                    'response_hash': self._hash_string(feedback.response),
                    'rating': feedback.rating,
                    'comments': feedback.comments,  # Keep comments as they may contain valuable insights
                    'timestamp': feedback.timestamp.isoformat(),
                    'query_metadata': feedback.query_metadata,
                    'processing_time': feedback.processing_time,
                    'similarity_scores': feedback.similarity_scores
                }
                anonymized_data.append(anonymized)
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'export_timestamp': datetime.now().isoformat(),
                    'total_entries': len(anonymized_data),
                    'feedback_data': anonymized_data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(anonymized_data)} anonymized feedback entries to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export anonymized feedback: {e}")
            raise
    
    def _hash_string(self, text: str) -> str:
        """
        Create a hash of the input string for anonymization.
        
        Args:
            text: Text to hash
            
        Returns:
            SHA-256 hash of the text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _read_all_feedback(self):
        """
        Generator to read all feedback entries from storage.
        
        Yields:
            FeedbackData objects
        """
        if not self.feedback_file.exists():
            return
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            yield FeedbackData.from_dict(data)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning(f"Skipping invalid feedback line: {e}")
                            continue
        except Exception as e:
            logger.error(f"Failed to read feedback file: {e}")
            raise
    
    def get_recent_feedback(self, days: int = 7) -> List[FeedbackData]:
        """
        Get feedback from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent feedback entries
        """
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        
        recent_feedback = []
        for feedback in self._read_all_feedback():
            if feedback.timestamp >= cutoff_date:
                recent_feedback.append(feedback)
        
        return sorted(recent_feedback, key=lambda x: x.timestamp, reverse=True)
    
    def flag_for_anomaly_detection(self, feedback_data: FeedbackData) -> Dict[str, Any]:
        """
        Flag feedback for anomaly detection if it meets certain criteria.
        
        Args:
            feedback_data: Feedback data to evaluate
            
        Returns:
            Dictionary containing flagging results and reasons
        """
        flagging_reasons = []
        should_flag = False
        
        # Flag low ratings (1-2) for anomaly detection
        if feedback_data.rating <= 2:
            flagging_reasons.append(f"low_rating_{feedback_data.rating}")
            should_flag = True
            logger.info(f"Flagging feedback {feedback_data.feedback_id} for anomaly detection (low rating: {feedback_data.rating})")
        
        # Flag if processing time is unusually high
        if feedback_data.processing_time > 10.0:  # More than 10 seconds
            flagging_reasons.append(f"high_processing_time_{feedback_data.processing_time:.2f}s")
            should_flag = True
            logger.info(f"Flagging feedback {feedback_data.feedback_id} for anomaly detection (high processing time: {feedback_data.processing_time}s)")
        
        # Flag if similarity scores are unusually low
        if feedback_data.similarity_scores and max(feedback_data.similarity_scores) < 0.3:
            flagging_reasons.append(f"low_similarity_scores_max_{max(feedback_data.similarity_scores):.3f}")
            should_flag = True
            logger.info(f"Flagging feedback {feedback_data.feedback_id} for anomaly detection (low similarity scores)")
        
        # Flag if comments contain negative keywords
        if feedback_data.comments:
            negative_keywords = ['wrong', 'incorrect', 'bad', 'terrible', 'useless', 'broken', 'error', 'fail']
            comment_lower = feedback_data.comments.lower()
            found_negative = [keyword for keyword in negative_keywords if keyword in comment_lower]
            if found_negative:
                flagging_reasons.append(f"negative_keywords_{','.join(found_negative)}")
                should_flag = True
                logger.info(f"Flagging feedback {feedback_data.feedback_id} for anomaly detection (negative keywords: {found_negative})")
        
        return {
            'should_flag': should_flag,
            'reasons': flagging_reasons,
            'feedback_id': feedback_data.feedback_id,
            'timestamp': datetime.now().isoformat(),
            'confidence_score': self._calculate_flagging_confidence(flagging_reasons)
        }
    
    def _calculate_flagging_confidence(self, reasons: List[str]) -> float:
        """
        Calculate confidence score for anomaly flagging based on reasons.
        
        Args:
            reasons: List of flagging reasons
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not reasons:
            return 0.0
        
        # Weight different types of reasons
        reason_weights = {
            'low_rating': 0.8,
            'high_processing_time': 0.6,
            'low_similarity_scores': 0.7,
            'negative_keywords': 0.5
        }
        
        total_weight = 0.0
        for reason in reasons:
            for reason_type, weight in reason_weights.items():
                if reason.startswith(reason_type):
                    total_weight += weight
                    break
        
        # Normalize to 0-1 range, with diminishing returns for multiple reasons
        confidence = min(total_weight / len(reasons), 1.0)
        return confidence
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        Analyze feedback patterns to identify potential system issues.
        
        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            feedback_list = list(self._read_all_feedback())
            
            if not feedback_list:
                return {'error': 'No feedback data available for analysis'}
            
            # Analyze rating trends over time
            rating_trends = self._analyze_rating_trends(feedback_list)
            
            # Analyze performance correlation
            performance_correlation = self._analyze_performance_correlation(feedback_list)
            
            # Identify problematic queries
            problematic_queries = self._identify_problematic_queries(feedback_list)
            
            # Analyze feedback-flagged items
            flagged_analysis = self._analyze_flagged_feedback(feedback_list)
            
            analysis_result = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_feedback_analyzed': len(feedback_list),
                'rating_trends': rating_trends,
                'performance_correlation': performance_correlation,
                'problematic_queries': problematic_queries,
                'flagged_analysis': flagged_analysis,
                'recommendations': self._generate_feedback_recommendations(
                    rating_trends, performance_correlation, problematic_queries, flagged_analysis
                )
            }
            
            logger.info(f"Completed feedback pattern analysis for {len(feedback_list)} entries")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {'error': str(e)}
    
    def _analyze_rating_trends(self, feedback_list: List[FeedbackData]) -> Dict[str, Any]:
        """Analyze rating trends over time"""
        # Group feedback by day
        daily_ratings = {}
        for feedback in feedback_list:
            date_key = feedback.timestamp.date().isoformat()
            if date_key not in daily_ratings:
                daily_ratings[date_key] = []
            daily_ratings[date_key].append(feedback.rating)
        
        # Calculate trends
        trend_data = {}
        for date, ratings in daily_ratings.items():
            trend_data[date] = {
                'average_rating': sum(ratings) / len(ratings),
                'count': len(ratings),
                'low_ratings_count': len([r for r in ratings if r <= 2]),
                'high_ratings_count': len([r for r in ratings if r >= 4])
            }
        
        # Identify declining trends
        dates = sorted(trend_data.keys())
        if len(dates) >= 3:
            recent_avg = sum(trend_data[date]['average_rating'] for date in dates[-3:]) / 3
            earlier_avg = sum(trend_data[date]['average_rating'] for date in dates[:3]) / 3
            trend_direction = 'declining' if recent_avg < earlier_avg - 0.5 else 'stable' if abs(recent_avg - earlier_avg) <= 0.5 else 'improving'
        else:
            trend_direction = 'insufficient_data'
        
        return {
            'daily_trends': trend_data,
            'trend_direction': trend_direction,
            'overall_average': sum(f.rating for f in feedback_list) / len(feedback_list)
        }
    
    def _analyze_performance_correlation(self, feedback_list: List[FeedbackData]) -> Dict[str, Any]:
        """Analyze correlation between performance metrics and ratings"""
        ratings = [f.rating for f in feedback_list]
        processing_times = [f.processing_time for f in feedback_list]
        
        # Calculate correlation between processing time and ratings
        if len(ratings) > 1 and len(processing_times) > 1:
            try:
                import numpy as np
                correlation = np.corrcoef(ratings, processing_times)[0, 1]
                correlation = float(correlation) if not np.isnan(correlation) else 0.0
            except ImportError:
                # Fallback calculation without numpy
                mean_rating = sum(ratings) / len(ratings)
                mean_time = sum(processing_times) / len(processing_times)
                
                numerator = sum((r - mean_rating) * (t - mean_time) for r, t in zip(ratings, processing_times))
                rating_var = sum((r - mean_rating) ** 2 for r in ratings)
                time_var = sum((t - mean_time) ** 2 for t in processing_times)
                
                correlation = numerator / (rating_var * time_var) ** 0.5 if rating_var > 0 and time_var > 0 else 0.0
        else:
            correlation = 0.0
        
        # Analyze similarity score correlation
        similarity_correlations = []
        for feedback in feedback_list:
            if feedback.similarity_scores:
                max_similarity = max(feedback.similarity_scores)
                similarity_correlations.append((feedback.rating, max_similarity))
        
        if similarity_correlations:
            sim_ratings, sim_scores = zip(*similarity_correlations)
            try:
                import numpy as np
                sim_correlation = np.corrcoef(sim_ratings, sim_scores)[0, 1]
                sim_correlation = float(sim_correlation) if not np.isnan(sim_correlation) else 0.0
            except ImportError:
                # Fallback calculation
                mean_sim_rating = sum(sim_ratings) / len(sim_ratings)
                mean_sim_score = sum(sim_scores) / len(sim_scores)
                
                numerator = sum((r - mean_sim_rating) * (s - mean_sim_score) for r, s in zip(sim_ratings, sim_scores))
                rating_var = sum((r - mean_sim_rating) ** 2 for r in sim_ratings)
                score_var = sum((s - mean_sim_score) ** 2 for s in sim_scores)
                
                sim_correlation = numerator / (rating_var * score_var) ** 0.5 if rating_var > 0 and score_var > 0 else 0.0
        else:
            sim_correlation = 0.0
        
        return {
            'processing_time_correlation': correlation,
            'similarity_score_correlation': sim_correlation,
            'average_processing_time': sum(processing_times) / len(processing_times),
            'performance_issues_detected': correlation < -0.3 or sim_correlation < 0.3
        }
    
    def _identify_problematic_queries(self, feedback_list: List[FeedbackData]) -> List[Dict[str, Any]]:
        """Identify queries that consistently receive poor ratings"""
        query_ratings = {}
        
        # Group ratings by query hash (for privacy)
        for feedback in feedback_list:
            query_hash = self._hash_string(feedback.query)
            if query_hash not in query_ratings:
                query_ratings[query_hash] = []
            query_ratings[query_hash].append({
                'rating': feedback.rating,
                'processing_time': feedback.processing_time,
                'similarity_scores': feedback.similarity_scores,
                'feedback_id': feedback.feedback_id
            })
        
        # Identify problematic queries (multiple low ratings)
        problematic = []
        for query_hash, ratings_data in query_ratings.items():
            if len(ratings_data) >= 2:  # At least 2 feedback entries
                avg_rating = sum(r['rating'] for r in ratings_data) / len(ratings_data)
                low_rating_count = len([r for r in ratings_data if r['rating'] <= 2])
                
                if avg_rating <= 2.5 or low_rating_count >= 2:
                    problematic.append({
                        'query_hash': query_hash,
                        'average_rating': avg_rating,
                        'total_feedback': len(ratings_data),
                        'low_rating_count': low_rating_count,
                        'average_processing_time': sum(r['processing_time'] for r in ratings_data) / len(ratings_data)
                    })
        
        return sorted(problematic, key=lambda x: x['average_rating'])
    
    def _analyze_flagged_feedback(self, feedback_list: List[FeedbackData]) -> Dict[str, Any]:
        """Analyze feedback that has been flagged for anomaly detection"""
        flagged_feedback = []
        
        for feedback in feedback_list:
            flagging_result = self.flag_for_anomaly_detection(feedback)
            if flagging_result['should_flag']:
                flagged_feedback.append({
                    'feedback_id': feedback.feedback_id,
                    'rating': feedback.rating,
                    'flagging_reasons': flagging_result['reasons'],
                    'confidence_score': flagging_result['confidence_score'],
                    'timestamp': feedback.timestamp.isoformat()
                })
        
        # Analyze flagging patterns
        reason_counts = {}
        for flagged in flagged_feedback:
            for reason in flagged['flagging_reasons']:
                reason_type = reason.split('_')[0] + '_' + reason.split('_')[1] if '_' in reason else reason
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        
        return {
            'total_flagged': len(flagged_feedback),
            'flagged_percentage': (len(flagged_feedback) / len(feedback_list)) * 100 if feedback_list else 0,
            'flagging_reasons_distribution': reason_counts,
            'flagged_entries': flagged_feedback[-10:]  # Last 10 flagged entries
        }
    
    def _generate_feedback_recommendations(self, rating_trends: Dict, performance_correlation: Dict, 
                                         problematic_queries: List, flagged_analysis: Dict) -> List[str]:
        """Generate recommendations based on feedback analysis"""
        recommendations = []
        
        # Rating trend recommendations
        if rating_trends['trend_direction'] == 'declining':
            recommendations.append("System performance is declining - investigate recent changes")
        
        # Performance correlation recommendations
        if performance_correlation['performance_issues_detected']:
            if performance_correlation['processing_time_correlation'] < -0.3:
                recommendations.append("High processing times correlate with low ratings - optimize performance")
            if performance_correlation['similarity_score_correlation'] < 0.3:
                recommendations.append("Low similarity scores correlate with poor ratings - review retrieval algorithm")
        
        # Problematic queries recommendations
        if len(problematic_queries) > 0:
            recommendations.append(f"Found {len(problematic_queries)} query patterns with consistently poor ratings - review content quality")
        
        # Flagged feedback recommendations
        if flagged_analysis['flagged_percentage'] > 20:
            recommendations.append("High percentage of flagged feedback - investigate potential system issues")
        
        if not recommendations:
            recommendations.append("System feedback patterns appear normal - continue monitoring")
        
        return recommendations