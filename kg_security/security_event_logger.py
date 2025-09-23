"""
Security Event Logger for Knowledge Graph Security

Implements comprehensive security event logging, audit trail management,
and integration with feedback system for anomaly flagging.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import sqlite3
import threading
from contextlib import contextmanager

from config import KG_CONFIG


@dataclass
class SecurityEvent:
    """Represents a security event in the system"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: str  # 'info', 'low', 'medium', 'high', 'critical'
    description: str
    node_id: Optional[str] = None
    user_id: Optional[str] = None
    source_component: Optional[str] = None
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AuditTrail:
    """Represents an audit trail entry"""
    trail_id: str
    timestamp: datetime
    action: str
    entity_type: str  # 'node', 'edge', 'graph'
    entity_id: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecurityEventLogger:
    """
    Manages security event logging, audit trails, and integration with feedback system
    """
    
    def __init__(self, config: Dict[str, Any] = None, db_path: str = "logs/security_events.db"):
        """
        Initialize the Security Event Logger
        
        Args:
            config: Configuration dictionary
            db_path: Path to SQLite database for persistent storage
        """
        self.config = config or KG_CONFIG
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for database operations
        self._db_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Event severity levels
        self.severity_levels = {
            'info': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        
        logger.info(f"SecurityEventLogger initialized with database: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for security events and audit trails"""
        try:
            with self._get_db_connection() as conn:
                # Create security events table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS security_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        description TEXT NOT NULL,
                        node_id TEXT,
                        user_id TEXT,
                        source_component TEXT,
                        metadata TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_timestamp TEXT,
                        resolution_notes TEXT
                    )
                ''')
                
                # Create audit trails table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_trails (
                        trail_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        action TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        user_id TEXT,
                        reason TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create indexes for better query performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON security_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_severity ON security_events(severity)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON security_events(event_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trails_timestamp ON audit_trails(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trails_entity ON audit_trails(entity_type, entity_id)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper locking"""
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def log_security_event(self, event_type: str, description: str, severity: str = 'info',
                          node_id: str = None, user_id: str = None, source_component: str = None,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Log a security event with timestamps and context
        
        Args:
            event_type: Type of security event
            description: Description of the event
            severity: Severity level (info, low, medium, high, critical)
            node_id: Optional node ID associated with the event
            user_id: Optional user ID who triggered the event
            source_component: Optional source component that generated the event
            metadata: Optional additional metadata
            
        Returns:
            Event ID of the logged event
        """
        try:
            event = SecurityEvent(
                event_id=f"{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                description=description,
                node_id=node_id,
                user_id=user_id,
                source_component=source_component,
                metadata=metadata or {}
            )
            
            # Store in database
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO security_events 
                    (event_id, timestamp, event_type, severity, description, node_id, 
                     user_id, source_component, metadata, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.severity,
                    event.description,
                    event.node_id,
                    event.user_id,
                    event.source_component,
                    json.dumps(event.metadata),
                    event.resolved
                ))
                conn.commit()
            
            # Log to structured logger
            logger.bind(
                event_id=event.event_id,
                event_type=event_type,
                severity=severity,
                node_id=node_id
            ).info(f"Security Event [{severity.upper()}]: {description}")
            
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")
            raise
    
    def log_audit_trail(self, action: str, entity_type: str, entity_id: str,
                       old_value: Dict[str, Any] = None, new_value: Dict[str, Any] = None,
                       user_id: str = None, reason: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Log an audit trail entry for tracking changes
        
        Args:
            action: Action performed (create, update, delete, quarantine, etc.)
            entity_type: Type of entity (node, edge, graph)
            entity_id: ID of the entity
            old_value: Previous value before change
            new_value: New value after change
            user_id: User who performed the action
            reason: Reason for the change
            metadata: Additional metadata
            
        Returns:
            Trail ID of the logged audit entry
        """
        try:
            trail = AuditTrail(
                trail_id=f"{action}_{entity_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                old_value=old_value,
                new_value=new_value,
                user_id=user_id,
                reason=reason,
                metadata=metadata or {}
            )
            
            # Store in database
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO audit_trails 
                    (trail_id, timestamp, action, entity_type, entity_id, old_value, 
                     new_value, user_id, reason, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trail.trail_id,
                    trail.timestamp.isoformat(),
                    trail.action,
                    trail.entity_type,
                    trail.entity_id,
                    json.dumps(trail.old_value) if trail.old_value else None,
                    json.dumps(trail.new_value) if trail.new_value else None,
                    trail.user_id,
                    trail.reason,
                    json.dumps(trail.metadata)
                ))
                conn.commit()
            
            logger.debug(f"Audit trail logged: {action} on {entity_type} {entity_id}")
            
            return trail.trail_id
            
        except Exception as e:
            logger.error(f"Error logging audit trail: {str(e)}")
            raise
    
    def get_security_events(self, limit: int = 100, severity_filter: str = None,
                           event_type_filter: str = None, start_time: datetime = None,
                           end_time: datetime = None, resolved_filter: bool = None) -> List[Dict[str, Any]]:
        """
        Retrieve security events with filtering options
        
        Args:
            limit: Maximum number of events to return
            severity_filter: Filter by severity level
            event_type_filter: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            resolved_filter: Filter by resolution status
            
        Returns:
            List of security events as dictionaries
        """
        try:
            query = "SELECT * FROM security_events WHERE 1=1"
            params = []
            
            if severity_filter:
                query += " AND severity = ?"
                params.append(severity_filter)
            
            if event_type_filter:
                query += " AND event_type = ?"
                params.append(event_type_filter)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if resolved_filter is not None:
                query += " AND resolved = ?"
                params.append(resolved_filter)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute(query, params)
                events = []
                
                for row in cursor.fetchall():
                    event_dict = dict(row)
                    # Parse JSON metadata
                    if event_dict['metadata']:
                        event_dict['metadata'] = json.loads(event_dict['metadata'])
                    else:
                        event_dict['metadata'] = {}
                    
                    events.append(event_dict)
                
                return events
                
        except Exception as e:
            logger.error(f"Error retrieving security events: {str(e)}")
            return []
    
    def get_audit_trails(self, limit: int = 100, entity_type_filter: str = None,
                        entity_id_filter: str = None, action_filter: str = None,
                        start_time: datetime = None, end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        Retrieve audit trails with filtering options
        
        Args:
            limit: Maximum number of trails to return
            entity_type_filter: Filter by entity type
            entity_id_filter: Filter by entity ID
            action_filter: Filter by action type
            start_time: Filter trails after this time
            end_time: Filter trails before this time
            
        Returns:
            List of audit trails as dictionaries
        """
        try:
            query = "SELECT * FROM audit_trails WHERE 1=1"
            params = []
            
            if entity_type_filter:
                query += " AND entity_type = ?"
                params.append(entity_type_filter)
            
            if entity_id_filter:
                query += " AND entity_id = ?"
                params.append(entity_id_filter)
            
            if action_filter:
                query += " AND action = ?"
                params.append(action_filter)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute(query, params)
                trails = []
                
                for row in cursor.fetchall():
                    trail_dict = dict(row)
                    # Parse JSON fields
                    for field in ['old_value', 'new_value', 'metadata']:
                        if trail_dict[field]:
                            trail_dict[field] = json.loads(trail_dict[field])
                        else:
                            trail_dict[field] = {} if field == 'metadata' else None
                    
                    trails.append(trail_dict)
                
                return trails
                
        except Exception as e:
            logger.error(f"Error retrieving audit trails: {str(e)}")
            return []
    
    def resolve_security_event(self, event_id: str, resolution_notes: str = None,
                              user_id: str = None) -> bool:
        """
        Mark a security event as resolved
        
        Args:
            event_id: ID of the event to resolve
            resolution_notes: Optional notes about the resolution
            user_id: User who resolved the event
            
        Returns:
            True if event was resolved successfully
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "UPDATE security_events SET resolved = ?, resolution_timestamp = ?, resolution_notes = ? WHERE event_id = ?",
                    (True, datetime.now().isoformat(), resolution_notes, event_id)
                )
                
                if cursor.rowcount > 0:
                    conn.commit()
                    
                    # Log audit trail for resolution
                    self.log_audit_trail(
                        action='resolve_event',
                        entity_type='security_event',
                        entity_id=event_id,
                        user_id=user_id,
                        reason='Event resolved',
                        metadata={'resolution_notes': resolution_notes}
                    )
                    
                    logger.info(f"Security event {event_id} resolved by {user_id or 'system'}")
                    return True
                else:
                    logger.warning(f"Security event {event_id} not found for resolution")
                    return False
                    
        except Exception as e:
            logger.error(f"Error resolving security event {event_id}: {str(e)}")
            return False
    
    def get_security_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get security metrics for the specified time window
        
        Args:
            time_window_hours: Time window in hours for metrics calculation
            
        Returns:
            Dictionary containing security metrics
        """
        try:
            start_time = datetime.now() - timedelta(hours=time_window_hours)
            
            with self._get_db_connection() as conn:
                # Count events by severity
                cursor = conn.execute('''
                    SELECT severity, COUNT(*) as count 
                    FROM security_events 
                    WHERE timestamp >= ? 
                    GROUP BY severity
                ''', (start_time.isoformat(),))
                
                severity_counts = {row['severity']: row['count'] for row in cursor.fetchall()}
                
                # Count events by type
                cursor = conn.execute('''
                    SELECT event_type, COUNT(*) as count 
                    FROM security_events 
                    WHERE timestamp >= ? 
                    GROUP BY event_type
                ''', (start_time.isoformat(),))
                
                event_type_counts = {row['event_type']: row['count'] for row in cursor.fetchall()}
                
                # Count resolved vs unresolved events
                cursor = conn.execute('''
                    SELECT resolved, COUNT(*) as count 
                    FROM security_events 
                    WHERE timestamp >= ? 
                    GROUP BY resolved
                ''', (start_time.isoformat(),))
                
                resolution_counts = {bool(row['resolved']): row['count'] for row in cursor.fetchall()}
                
                # Get total event count
                cursor = conn.execute('''
                    SELECT COUNT(*) as total 
                    FROM security_events 
                    WHERE timestamp >= ?
                ''', (start_time.isoformat(),))
                
                total_events = cursor.fetchone()['total']
                
                metrics = {
                    'time_window_hours': time_window_hours,
                    'total_events': total_events,
                    'events_by_severity': severity_counts,
                    'events_by_type': event_type_counts,
                    'resolved_events': resolution_counts.get(True, 0),
                    'unresolved_events': resolution_counts.get(False, 0),
                    'resolution_rate': resolution_counts.get(True, 0) / total_events if total_events > 0 else 0,
                    'critical_events': severity_counts.get('critical', 0),
                    'high_severity_events': severity_counts.get('high', 0) + severity_counts.get('critical', 0)
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating security metrics: {str(e)}")
            return {}
    
    def integrate_with_feedback_system(self, feedback_data: Dict[str, Any]) -> None:
        """
        Integrate with feedback system for anomaly flagging
        
        Args:
            feedback_data: Feedback data containing ratings and comments
        """
        try:
            # Check if feedback indicates potential security issues
            rating = feedback_data.get('rating', 5)
            comments = feedback_data.get('comments', '').lower()
            
            # Flag low ratings as potential security concerns
            if rating <= 2:
                self.log_security_event(
                    event_type='low_rating_feedback',
                    description=f"Low rating ({rating}/5) received for query response",
                    severity='low',
                    metadata={
                        'feedback_id': feedback_data.get('feedback_id'),
                        'query': feedback_data.get('query', ''),
                        'rating': rating,
                        'comments': feedback_data.get('comments', '')
                    }
                )
            
            # Check for security-related keywords in comments
            security_keywords = ['suspicious', 'wrong', 'incorrect', 'fake', 'malicious', 'tampered', 'corrupted']
            if any(keyword in comments for keyword in security_keywords):
                self.log_security_event(
                    event_type='security_concern_feedback',
                    description=f"Feedback contains security-related concerns: {comments[:100]}",
                    severity='medium',
                    metadata={
                        'feedback_id': feedback_data.get('feedback_id'),
                        'query': feedback_data.get('query', ''),
                        'rating': rating,
                        'comments': feedback_data.get('comments', ''),
                        'detected_keywords': [kw for kw in security_keywords if kw in comments]
                    }
                )
            
        except Exception as e:
            logger.error(f"Error integrating with feedback system: {str(e)}")
    
    def export_security_report(self, output_path: Path, start_time: datetime = None,
                              end_time: datetime = None) -> bool:
        """
        Export comprehensive security report to file
        
        Args:
            output_path: Path to save the report
            start_time: Start time for report data
            end_time: End time for report data
            
        Returns:
            True if report was exported successfully
        """
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)  # Last 30 days
            if not end_time:
                end_time = datetime.now()
            
            # Get security events and audit trails
            events = self.get_security_events(
                limit=10000,
                start_time=start_time,
                end_time=end_time
            )
            
            trails = self.get_audit_trails(
                limit=10000,
                start_time=start_time,
                end_time=end_time
            )
            
            # Get metrics
            metrics = self.get_security_metrics(time_window_hours=24 * 30)  # 30 days
            
            # Create comprehensive report
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_events': len(events),
                    'total_audit_trails': len(trails)
                },
                'security_metrics': metrics,
                'security_events': events,
                'audit_trails': trails
            }
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Security report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting security report: {str(e)}")
            return False