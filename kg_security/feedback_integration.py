"""
Feedback Integration Module for Knowledge Graph Security

Provides integration between the feedback system and knowledge graph anomaly detection,
enabling feedback-driven security analysis and correlation detection.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger
import json
import uuid
from pathlib import Path


@dataclass
class FeedbackSecurityEvent:
    """Represents a security event generated from feedback analysis"""
    event_id: str
    event_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    feedback_data: Dict[str, Any]
    correlation_data: Optional[Dict[str, Any]] = None
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class FeedbackSecurityIntegrator:
    """
    Integrates feedback system analysis with knowledge graph security layer.
    Provides correlation analysis and feedback-driven anomaly detection.
    """
    
    def __init__(self, kg_manager=None, feedback_system=None):
        """
        Initialize the feedback security integrator.
        
        Args:
            kg_manager: KnowledgeGraphManager instance
            feedback_system: FeedbackSystem instance
        """
        self.kg_manager = kg_manager
        self.feedback_system = feedback_system
        self.integration_events = []
        
        logger.info("FeedbackSecurityIntegrator initialized")
    
    def analyze_feedback_security_correlation(self) -> Dict[str, Any]:
        """
        Analyze correlation between feedback patterns and security events.
        
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            if not self.feedback_system:
                logger.warning("No feedback system available for correlation analysis")
                return {'error': 'No feedback system available'}
            
            # Get feedback pattern analysis
            feedback_analysis = self.feedback_system.analyze_feedback_patterns()
            
            if 'error' in feedback_analysis:
                return feedback_analysis
            
            # Integrate with knowledge graph security layer
            if self.kg_manager:
                security_events = self.kg_manager.integrate_feedback_analysis(feedback_analysis)
                
                # Get flagged feedback items for detailed correlation
                flagged_items = feedback_analysis.get('flagged_analysis', {}).get('flagged_entries', [])
                
                if flagged_items:
                    correlation_result = self.kg_manager.correlate_feedback_with_graph_anomalies(flagged_items)
                else:
                    correlation_result = {'message': 'No flagged feedback items for correlation'}
                
                # Generate security events from correlation
                correlation_events = self._generate_correlation_security_events(correlation_result)
                
                analysis_result = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'feedback_analysis': feedback_analysis,
                    'security_events_generated': len(security_events),
                    'correlation_analysis': correlation_result,
                    'correlation_events': correlation_events,
                    'integration_summary': self._generate_integration_summary(
                        feedback_analysis, security_events, correlation_result
                    )
                }
                
                logger.info(f"Completed feedback-security correlation analysis with {len(security_events)} security events")
                return analysis_result
            else:
                logger.warning("No knowledge graph manager available for security integration")
                return {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'feedback_analysis': feedback_analysis,
                    'security_integration': 'unavailable',
                    'message': 'Knowledge graph manager not available'
                }
                
        except Exception as e:
            logger.error(f"Error in feedback-security correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def _generate_correlation_security_events(self, correlation_result: Dict[str, Any]) -> List[FeedbackSecurityEvent]:
        """Generate security events from correlation analysis"""
        events = []
        
        try:
            if 'error' in correlation_result:
                return events
            
            correlation_insights = correlation_result.get('correlation_insights', [])
            
            for insight in correlation_insights:
                if insight.get('severity') in ['medium', 'high']:
                    event = FeedbackSecurityEvent(
                        event_id=f"correlation_{uuid.uuid4().hex[:8]}",
                        event_type=f"feedback_correlation_{insight.get('type', 'unknown')}",
                        description=insight.get('description', 'Correlation detected'),
                        severity=insight.get('severity', 'low'),
                        timestamp=datetime.now(),
                        feedback_data={
                            'correlation_strength': insight.get('confidence', 0),
                            'insight_type': insight.get('type'),
                            'details': insight.get('details', '')
                        },
                        correlation_data=correlation_result,
                        recommended_actions=self._get_recommended_actions_for_insight(insight)
                    )
                    events.append(event)
                    self.integration_events.append(event)
            
            # Generate high-level correlation event if strong correlation detected
            temporal_strength = correlation_result.get('temporal_correlation', {}).get('correlation_strength', 0)
            content_strength = correlation_result.get('content_correlation', {}).get('correlation_strength', 0)
            
            if temporal_strength > 0.5 or content_strength > 0.3:
                event = FeedbackSecurityEvent(
                    event_id=f"high_correlation_{uuid.uuid4().hex[:8]}",
                    event_type="high_feedback_anomaly_correlation",
                    description=f"Strong correlation detected between feedback and graph anomalies",
                    severity='medium',
                    timestamp=datetime.now(),
                    feedback_data={
                        'temporal_correlation': temporal_strength,
                        'content_correlation': content_strength
                    },
                    correlation_data=correlation_result,
                    recommended_actions=[
                        "Investigate root cause of correlated issues",
                        "Review system performance and data quality",
                        "Consider implementing additional monitoring"
                    ]
                )
                events.append(event)
                self.integration_events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error generating correlation security events: {str(e)}")
            return events
    
    def _get_recommended_actions_for_insight(self, insight: Dict[str, Any]) -> List[str]:
        """Get recommended actions based on insight type"""
        insight_type = insight.get('type', '')
        severity = insight.get('severity', 'low')
        
        actions = []
        
        if insight_type == 'temporal_correlation':
            actions.extend([
                "Investigate temporal patterns in system behavior",
                "Review recent system changes or deployments",
                "Monitor for recurring temporal patterns"
            ])
        
        elif insight_type == 'content_correlation':
            actions.extend([
                "Review content quality and relevance",
                "Investigate data source integrity",
                "Consider content filtering improvements"
            ])
        
        elif insight_type == 'high_flagging_rate':
            actions.extend([
                "Review system performance metrics",
                "Investigate user experience issues",
                "Consider system optimization"
            ])
        
        elif insight_type == 'centrality_anomalies':
            actions.extend([
                "Review knowledge graph structure",
                "Investigate node relationship patterns",
                "Consider graph pruning or restructuring"
            ])
        
        elif insight_type == 'suspicious_content_pattern':
            actions.extend([
                "URGENT: Conduct immediate security review",
                "Isolate suspicious content",
                "Review access logs and user activity"
            ])
        
        if severity == 'high':
            actions.insert(0, "PRIORITY: Immediate investigation required")
        
        return actions
    
    def _generate_integration_summary(self, feedback_analysis: Dict, security_events: List, 
                                    correlation_result: Dict) -> Dict[str, Any]:
        """Generate summary of integration results"""
        try:
            flagged_analysis = feedback_analysis.get('flagged_analysis', {})
            
            summary = {
                'total_feedback_analyzed': feedback_analysis.get('total_feedback_analyzed', 0),
                'flagged_feedback_percentage': flagged_analysis.get('flagged_percentage', 0),
                'security_events_generated': len(security_events),
                'correlation_strength': {
                    'temporal': correlation_result.get('temporal_correlation', {}).get('correlation_strength', 0),
                    'content': correlation_result.get('content_correlation', {}).get('correlation_strength', 0)
                },
                'risk_level': self._assess_risk_level(feedback_analysis, security_events, correlation_result),
                'key_insights': [event.get('description', '') for event in security_events[:3]],  # Top 3 insights
                'recommended_priority_actions': self._get_priority_actions(security_events)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating integration summary: {str(e)}")
            return {'error': str(e)}
    
    def _assess_risk_level(self, feedback_analysis: Dict, security_events: List, correlation_result: Dict) -> str:
        """Assess overall risk level based on analysis results"""
        try:
            risk_score = 0
            
            # Factor in flagged feedback percentage
            flagged_percentage = feedback_analysis.get('flagged_analysis', {}).get('flagged_percentage', 0)
            if flagged_percentage > 30:
                risk_score += 3
            elif flagged_percentage > 15:
                risk_score += 2
            elif flagged_percentage > 5:
                risk_score += 1
            
            # Factor in security events
            high_severity_events = sum(1 for event in security_events if event.get('severity') == 'high')
            medium_severity_events = sum(1 for event in security_events if event.get('severity') == 'medium')
            
            risk_score += high_severity_events * 3
            risk_score += medium_severity_events * 2
            
            # Factor in correlation strength
            temporal_strength = correlation_result.get('temporal_correlation', {}).get('correlation_strength', 0)
            content_strength = correlation_result.get('content_correlation', {}).get('correlation_strength', 0)
            
            if temporal_strength > 0.7 or content_strength > 0.5:
                risk_score += 3
            elif temporal_strength > 0.4 or content_strength > 0.3:
                risk_score += 2
            elif temporal_strength > 0.2 or content_strength > 0.1:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 8:
                return 'critical'
            elif risk_score >= 5:
                return 'high'
            elif risk_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {str(e)}")
            return 'unknown'
    
    def _get_priority_actions(self, security_events: List[FeedbackSecurityEvent]) -> List[str]:
        """Get priority actions from security events"""
        priority_actions = []
        
        # Collect all recommended actions from high and medium severity events
        for event in security_events:
            if event.get('severity') in ['high', 'medium']:
                priority_actions.extend(event.get('recommended_actions', []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in priority_actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return unique_actions[:5]  # Return top 5 priority actions
    
    def export_integration_report(self, output_path: Path) -> bool:
        """Export integration analysis report"""
        try:
            analysis_result = self.analyze_feedback_security_correlation()
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'feedback_security_integration',
                    'version': '1.0'
                },
                'analysis_result': analysis_result,
                'integration_events': [event.to_dict() for event in self.integration_events]
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Integration report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting integration report: {str(e)}")
            return False
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get metrics about the integration process"""
        try:
            return {
                'total_integration_events': len(self.integration_events),
                'events_by_severity': {
                    'low': sum(1 for e in self.integration_events if e.severity == 'low'),
                    'medium': sum(1 for e in self.integration_events if e.severity == 'medium'),
                    'high': sum(1 for e in self.integration_events if e.severity == 'high'),
                    'critical': sum(1 for e in self.integration_events if e.severity == 'critical')
                },
                'events_by_type': self._count_events_by_type(),
                'latest_integration': self.integration_events[-1].timestamp.isoformat() if self.integration_events else None
            }
        except Exception as e:
            logger.error(f"Error getting integration metrics: {str(e)}")
            return {}
    
    def _count_events_by_type(self) -> Dict[str, int]:
        """Count integration events by type"""
        type_counts = {}
        for event in self.integration_events:
            event_type = event.event_type
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        return type_counts