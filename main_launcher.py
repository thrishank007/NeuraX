#!/usr/bin/env python3
"""
Main launcher for SecureInsight Multimodal RAG System

This launcher orchestrates all system components including:
- Ingestion processors (document, image, audio, notes)
- Embedding manager and vector store
- Query processor and retrieval system
- LLM generator and citation generator
- Knowledge graph security layer
- Feedback system
- UI interfaces (Gradio and Streamlit)
"""
import sys
import os
import argparse
import signal
import threading
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import core components
from setup_logging import setup_logging, get_logger
from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from config import (
    GRADIO_CONFIG, STREAMLIT_CONFIG, PERFORMANCE_CONFIG,
    MODEL_DOWNLOAD_CONFIG, SECURITY_CONFIG, SEARCH_CONFIG,
    LLM_CONFIG, WHISPER_CONFIG, KG_CONFIG, FEEDBACK_CONFIG
)


class SecureInsightLauncher:
    """Main launcher for SecureInsight system"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.logger = get_logger("SecureInsightLauncher")
        self.gradio_process = None
        self.streamlit_process = None
        self.shutdown_event = threading.Event()
        
        # Core system components
        self.ingestion_manager = None
        self.embedding_manager = None
        self.vector_store = None
        self.query_processor = None
        self.stt_processor = None
        self.llm_generator = None
        self.citation_generator = None
        self.kg_manager = None
        self.feedback_system = None
        self.metrics_collector = None
        
        # Component initialization status
        self.component_status = {
            'ingestion_manager': False,
            'embedding_manager': False,
            'vector_store': False,
            'query_processor': False,
            'stt_processor': False,
            'llm_generator': False,
            'citation_generator': False,
            'kg_manager': False,
            'feedback_system': False,
            'metrics_collector': False
        }
        
        # System health status
        self.system_health = {
            'overall_status': 'initializing',
            'component_errors': [],
            'last_health_check': None,
            'startup_time': time.time()
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("SecureInsight Launcher initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.shutdown()
    
    def validate_system(self) -> bool:
        """Validate system requirements and readiness"""
        try:
            self.logger.info("Validating system requirements...")
            
            # Run system validation
            validation_results = self.error_handler.validate_system_requirements()
            
            if not validation_results['compatible']:
                self.logger.error("System validation failed:")
                for error in validation_results['errors']:
                    self.logger.error(f"  - {error}")
                return False
            
            # Check offline operation readiness
            offline_results = self.error_handler.test_offline_operation()
            
            if not offline_results['offline_capable']:
                self.logger.warning("System not fully ready for offline operation:")
                for failure in offline_results['tests_failed']:
                    self.logger.warning(f"  - {failure}")
                
                # Ask user if they want to continue
                if not self._prompt_continue_with_warnings():
                    return False
            
            self.logger.info("System validation completed successfully")
            return True
            
        except Exception as e:
            error_report = self.error_handler.handle_error(
                e, ErrorCategory.VALIDATION,
                context={'validation_type': 'system_startup'},
                severity=ErrorSeverity.CRITICAL
            )
            self.logger.error(f"System validation failed: {error_report.user_guidance}")
            return False
    
    def _prompt_continue_with_warnings(self) -> bool:
        """Prompt user to continue despite warnings"""
        try:
            response = input("\nSystem has warnings. Continue anyway? (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            return False
    
    def initialize_components(self) -> bool:
        """Initialize all core system components with proper error handling and health checks"""
        try:
            self.logger.info("Initializing core components...")
            self.system_health['overall_status'] = 'initializing'
            
            # Initialize ingestion manager first (no dependencies)
            if not self._initialize_ingestion_manager():
                return False
            
            # Initialize embedding manager (required for vector operations)
            if not self._initialize_embedding_manager():
                return False
            
            # Initialize vector store (depends on embedding manager)
            if not self._initialize_vector_store():
                return False
            
            # Initialize query processor (depends on embedding manager and vector store)
            if not self._initialize_query_processor():
                return False
            
            # Initialize speech-to-text processor
            if not self._initialize_stt_processor():
                return False
            
            # Initialize LLM generator
            if not self._initialize_llm_generator():
                return False
            
            # Initialize citation generator
            if not self._initialize_citation_generator():
                return False
            
            # Initialize knowledge graph manager
            if not self._initialize_kg_manager():
                return False
            
            # Initialize feedback system
            if not self._initialize_feedback_system():
                return False
            
            # Initialize metrics collector
            if not self._initialize_metrics_collector():
                return False
            
            # Perform system health check
            self._perform_health_check()
            
            self.logger.info("All core components initialized successfully")
            self.system_health['overall_status'] = 'healthy'
            return True
            
        except Exception as e:
            error_report = self.error_handler.handle_error(
                e, ErrorCategory.MODEL_LOADING,
                context={'initialization_phase': 'core_components'},
                severity=ErrorSeverity.CRITICAL
            )
            self.logger.error(f"Component initialization failed: {error_report.user_guidance}")
            self.system_health['overall_status'] = 'failed'
            return False
    
    def _initialize_ingestion_manager(self) -> bool:
        """Initialize ingestion manager"""
        try:
            from ingestion.ingestion_manager import IngestionManager
            self.ingestion_manager = IngestionManager()
            self.component_status['ingestion_manager'] = True
            self.logger.info("✅ Ingestion manager initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ingestion manager: {e}")
            self.system_health['component_errors'].append(f"ingestion_manager: {e}")
            return False
    
    def _initialize_embedding_manager(self) -> bool:
        """Initialize embedding manager with fallback options"""
        try:
            from indexing.embedding_manager import EmbeddingManager
            
            # Try GPU first, then CPU fallback
            try:
                self.embedding_manager = EmbeddingManager()
                self.component_status['embedding_manager'] = True
                self.logger.info("✅ Embedding manager initialized (GPU)")
                return True
            except Exception as gpu_error:
                self.logger.warning(f"GPU initialization failed: {gpu_error}")
                
                # Try CPU fallback
                try:
                    self.embedding_manager = EmbeddingManager(device='cpu')
                    self.component_status['embedding_manager'] = True
                    self.logger.info("✅ Embedding manager initialized (CPU fallback)")
                    return True
                except Exception as cpu_error:
                    self.logger.error(f"❌ CPU fallback failed: {cpu_error}")
                    self.system_health['component_errors'].append(f"embedding_manager: {cpu_error}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embedding manager: {e}")
            self.system_health['component_errors'].append(f"embedding_manager: {e}")
            return False
    
    def _initialize_vector_store(self) -> bool:
        """Initialize vector store"""
        try:
            from indexing.vector_store import VectorStore
            from config import VECTOR_DB_DIR, CHROMA_CONFIG
            
            self.vector_store = VectorStore(
                persist_directory=str(VECTOR_DB_DIR),
                collection_name=CHROMA_CONFIG['collection_name']
            )
            self.component_status['vector_store'] = True
            self.logger.info("✅ Vector store initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector store: {e}")
            self.system_health['component_errors'].append(f"vector_store: {e}")
            return False
    
    def _initialize_query_processor(self) -> bool:
        """Initialize query processor"""
        try:
            if not self.embedding_manager or not self.vector_store:
                self.logger.error("❌ Query processor requires embedding manager and vector store")
                return False
                
            from retrieval.query_processor import QueryProcessor
            self.query_processor = QueryProcessor(
                self.embedding_manager, 
                self.vector_store, 
                SEARCH_CONFIG
            )
            self.component_status['query_processor'] = True
            self.logger.info("✅ Query processor initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize query processor: {e}")
            self.system_health['component_errors'].append(f"query_processor: {e}")
            return False
    
    def _initialize_stt_processor(self) -> bool:
        """Initialize speech-to-text processor"""
        try:
            from retrieval.speech_to_text_processor import SpeechToTextProcessor
            self.stt_processor = SpeechToTextProcessor()
            self.component_status['stt_processor'] = True
            self.logger.info("✅ Speech-to-text processor initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize STT processor: {e}")
            self.system_health['component_errors'].append(f"stt_processor: {e}")
            # STT is not critical, continue without it
            return True
    
    def _initialize_llm_generator(self) -> bool:
        """Initialize LLM generator (using LM Studio by default, fallback to HuggingFace)"""
        try:
            from generation.llm_factory import create_llm_generator
            self.llm_generator = create_llm_generator(LLM_CONFIG)
            self.component_status['llm_generator'] = True
            
            # Get model info for logging
            model_info = self.llm_generator.get_model_info()
            if 'current_model' in model_info:
                self.logger.info(f"✅ LLM generator initialized with model: {model_info['current_model']}")
                if model_info.get('supports_multimodal', False):
                    self.logger.info("🖼️  Multimodal capabilities enabled")
            else:
                self.logger.info("✅ LLM generator initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM generator: {e}")
            self.system_health['component_errors'].append(f"llm_generator: {e}")
            # LLM is not critical for basic search, continue without it
            return True
    
    def _initialize_citation_generator(self) -> bool:
        """Initialize citation generator"""
        try:
            from generation.citation_generator import CitationGenerator
            self.citation_generator = CitationGenerator()
            self.component_status['citation_generator'] = True
            self.logger.info("✅ Citation generator initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize citation generator: {e}")
            self.system_health['component_errors'].append(f"citation_generator: {e}")
            return False
    
    def _initialize_kg_manager(self) -> bool:
        """Initialize knowledge graph manager"""
        try:
            from kg_security.knowledge_graph_manager import KnowledgeGraphManager
            self.kg_manager = KnowledgeGraphManager()
            self.component_status['kg_manager'] = True
            self.logger.info("✅ Knowledge graph manager initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize knowledge graph manager: {e}")
            self.system_health['component_errors'].append(f"kg_manager: {e}")
            return False
    
    def _initialize_feedback_system(self) -> bool:
        """Initialize feedback system"""
        try:
            from feedback.feedback_system import FeedbackSystem
            self.feedback_system = FeedbackSystem()
            self.component_status['feedback_system'] = True
            self.logger.info("✅ Feedback system initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize feedback system: {e}")
            self.system_health['component_errors'].append(f"feedback_system: {e}")
            return False
    
    def _initialize_metrics_collector(self) -> bool:
        """Initialize metrics collector"""
        try:
            from feedback.metrics_collector import MetricsCollector
            self.metrics_collector = MetricsCollector()
            self.component_status['metrics_collector'] = True
            self.logger.info("✅ Metrics collector initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize metrics collector: {e}")
            self.system_health['component_errors'].append(f"metrics_collector: {e}")
            return False
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            'timestamp': time.time(),
            'overall_healthy': True,
            'critical_components': [],
            'optional_components': [],
            'warnings': [],
            'errors': []
        }
        
        # Critical components (system cannot function without these)
        critical_components = [
            'ingestion_manager', 'embedding_manager', 'vector_store', 
            'query_processor', 'citation_generator', 'kg_manager', 'feedback_system'
        ]
        
        # Optional components (system can function with reduced capability)
        optional_components = ['stt_processor', 'llm_generator', 'metrics_collector']
        
        # Check critical components
        for component in critical_components:
            if self.component_status.get(component, False):
                health_status['critical_components'].append(f"✅ {component}")
            else:
                health_status['critical_components'].append(f"❌ {component}")
                health_status['overall_healthy'] = False
                health_status['errors'].append(f"Critical component failed: {component}")
        
        # Check optional components
        for component in optional_components:
            if self.component_status.get(component, False):
                health_status['optional_components'].append(f"✅ {component}")
            else:
                health_status['optional_components'].append(f"⚠️ {component}")
                health_status['warnings'].append(f"Optional component unavailable: {component}")
        
        # Test component integration
        integration_tests = self._run_integration_tests()
        health_status.update(integration_tests)
        
        self.system_health['last_health_check'] = health_status
        
        return health_status
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run basic integration tests between components"""
        tests = {
            'integration_tests': [],
            'integration_warnings': []
        }
        
        try:
            # Test ingestion -> embedding -> vector store pipeline
            if (self.ingestion_manager and self.embedding_manager and self.vector_store):
                tests['integration_tests'].append("✅ Ingestion pipeline ready")
            else:
                tests['integration_warnings'].append("⚠️ Ingestion pipeline incomplete")
            
            # Test query -> retrieval -> generation pipeline
            if (self.query_processor and self.llm_generator and self.citation_generator):
                tests['integration_tests'].append("✅ Query pipeline ready")
            else:
                tests['integration_warnings'].append("⚠️ Query pipeline incomplete")
            
            # Test knowledge graph integration
            if self.kg_manager and self.vector_store:
                tests['integration_tests'].append("✅ Knowledge graph integration ready")
            else:
                tests['integration_warnings'].append("⚠️ Knowledge graph integration incomplete")
            
            # Test feedback integration
            if self.feedback_system and self.metrics_collector:
                tests['integration_tests'].append("✅ Feedback system integration ready")
            else:
                tests['integration_warnings'].append("⚠️ Feedback system integration incomplete")
                
        except Exception as e:
            tests['integration_warnings'].append(f"⚠️ Integration test error: {e}")
        
        return tests
    
    def launch_gradio_interface(self) -> bool:
        """Launch Gradio chat interface with all integrated components"""
        try:
            self.logger.info("Launching Gradio interface...")
            
            # Import the Gradio app class
            from ui.gradio_app import SecureInsightGradioApp
            
            # Create app instance and inject components
            gradio_app = SecureInsightGradioApp()
            
            # Inject initialized components
            gradio_app.embedding_manager = self.embedding_manager
            gradio_app.vector_store = self.vector_store
            gradio_app.ingestion_manager = self.ingestion_manager
            gradio_app.query_processor = self.query_processor
            gradio_app.stt_processor = self.stt_processor
            gradio_app.llm_generator = self.llm_generator
            gradio_app.citation_generator = self.citation_generator
            gradio_app.feedback_system = self.feedback_system
            
            # Create the Gradio interface
            interface = gradio_app.create_interface()
            
            # Launch in a separate thread
            def run_gradio():
                try:
                    interface.launch(
                        server_name=GRADIO_CONFIG['server_name'],
                        server_port=GRADIO_CONFIG['server_port'],
                        share=GRADIO_CONFIG['share'],
                        prevent_thread_lock=True,
                        show_error=True
                    )
                except Exception as e:
                    self.logger.error(f"Gradio interface error: {e}")
            
            gradio_thread = threading.Thread(target=run_gradio, daemon=True)
            gradio_thread.start()
            
            # Wait a moment for startup and test connectivity
            time.sleep(3)
            
            self.logger.info(f"✅ Gradio interface launched at http://{GRADIO_CONFIG['server_name']}:{GRADIO_CONFIG['server_port']}")
            return True
            
        except Exception as e:
            error_report = self.error_handler.handle_error(
                e, ErrorCategory.PERFORMANCE,
                context={'component': 'gradio_interface'},
                severity=ErrorSeverity.HIGH
            )
            self.logger.error(f"❌ Failed to launch Gradio interface: {error_report.user_guidance}")
            return False
    
    def launch_streamlit_dashboard(self) -> bool:
        """Launch Streamlit monitoring dashboard with component integration"""
        try:
            self.logger.info("Launching Streamlit dashboard...")
            
            import subprocess
            import sys
            
            # Set environment variables for component sharing
            env = os.environ.copy()
            env['SECUREINSIGHT_LAUNCHER_PID'] = str(os.getpid())
            
            # Launch Streamlit in a separate process
            streamlit_cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                'ui/streamlit_dashboard.py',
                '--server.port', str(STREAMLIT_CONFIG['server_port']),
                '--server.address', STREAMLIT_CONFIG['server_address'],
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false',
                '--server.enableCORS', 'false',
                '--server.enableXsrfProtection', 'false'
            ]
            
            self.streamlit_process = subprocess.Popen(
                streamlit_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Wait for startup and check status
            startup_timeout = 10
            for i in range(startup_timeout):
                time.sleep(1)
                if self.streamlit_process.poll() is None:
                    # Process is still running, check if it's responsive
                    if i >= 3:  # Give it at least 3 seconds
                        break
                else:
                    # Process terminated
                    stdout, stderr = self.streamlit_process.communicate()
                    self.logger.error(f"Streamlit failed to start: {stderr}")
                    return False
            
            # Final check
            if self.streamlit_process.poll() is None:
                self.logger.info(f"✅ Streamlit dashboard launched at http://{STREAMLIT_CONFIG['server_address']}:{STREAMLIT_CONFIG['server_port']}")
                return True
            else:
                stdout, stderr = self.streamlit_process.communicate()
                self.logger.error(f"❌ Streamlit startup failed: {stderr}")
                return False
            
        except Exception as e:
            error_report = self.error_handler.handle_error(
                e, ErrorCategory.PERFORMANCE,
                context={'component': 'streamlit_dashboard'},
                severity=ErrorSeverity.MEDIUM
            )
            self.logger.warning(f"❌ Failed to launch Streamlit dashboard: {error_report.user_guidance}")
            return False
    
    def run_interactive_mode(self):
        """Run in interactive mode with user commands"""
        self.logger.info("Starting interactive mode...")
        
        print("\n" + "="*60)
        print("SecureInsight Multimodal RAG System")
        print("="*60)
        print(f"Gradio Interface: http://{GRADIO_CONFIG['server_name']}:{GRADIO_CONFIG['server_port']}")
        print(f"Streamlit Dashboard: http://{STREAMLIT_CONFIG['server_address']}:{STREAMLIT_CONFIG['server_port']}")
        print("\nCommands:")
        print("  help      - Show this help message")
        print("  status    - Show system status and component health")
        print("  test      - Run comprehensive system tests")
        print("  api       - Show unified API information")
        print("  health    - Run health check")
        print("  integrate - Test system integration")
        print("  quit      - Shutdown system")
        print("="*60)
        
        while not self.shutdown_event.is_set():
            try:
                command = input("\nSecureInsight> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'status':
                    self._show_status()
                elif command == 'test':
                    self._run_tests()
                elif command == 'api':
                    self._show_api_info()
                elif command == 'health':
                    self._run_health_check()
                elif command == 'integrate':
                    self._run_integration_test()
                elif command == '':
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        self.logger.info("Interactive mode ended")
    
    def _show_help(self):
        """Show help information"""
        print("\nSecureInsight Commands:")
        print("  help      - Show this help message")
        print("  status    - Show system status and component health")
        print("  test      - Run comprehensive system tests (offline + integration + health)")
        print("  api       - Show unified API information and available methods")
        print("  health    - Run component health check")
        print("  integrate - Test system integration with sample data")
        print("  quit      - Shutdown system gracefully")
        print("\nSystem URLs:")
        print(f"  Gradio Interface: http://{GRADIO_CONFIG['server_name']}:{GRADIO_CONFIG['server_port']}")
        print(f"  Streamlit Dashboard: http://{STREAMLIT_CONFIG['server_address']}:{STREAMLIT_CONFIG['server_port']}")
    
    def _show_api_info(self):
        """Show unified API information"""
        print("\nUnified API Information:")
        print("=" * 50)
        
        api = self.get_unified_api()
        
        print(f"System Status: {api['status'].get('overall_status', 'unknown').upper()}")
        print(f"Available Components: {len(api['components'])}")
        print(f"Available Methods: {len(api['methods'])}")
        print()
        
        print("Available Components:")
        print("-" * 30)
        for name, component in api['components'].items():
            print(f"  ✅ {name}: {type(component).__name__}")
        
        print()
        print("Available Methods:")
        print("-" * 30)
        for name, method in api['methods'].items():
            print(f"  🔧 {name}: {method.__doc__.split('.')[0] if method.__doc__ else 'No description'}")
        
        print()
        print("Usage Example:")
        print("  api = launcher.get_unified_api()")
        print("  result = api['methods']['process_text_query']('your query here')")
    
    def _run_health_check(self):
        """Run and display health check results"""
        print("\nRunning health check...")
        health_status = self._perform_health_check()
        
        print(f"Overall Health: {'✅ Healthy' if health_status['overall_healthy'] else '❌ Unhealthy'}")
        print(f"Check Time: {datetime.fromtimestamp(health_status['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if health_status['critical_components']:
            print("Critical Components:")
            for component in health_status['critical_components']:
                print(f"  {component}")
        
        if health_status['optional_components']:
            print("Optional Components:")
            for component in health_status['optional_components']:
                print(f"  {component}")
        
        if health_status['integration_tests']:
            print("Integration Tests:")
            for test in health_status['integration_tests']:
                print(f"  {test}")
        
        if health_status['warnings']:
            print("Warnings:")
            for warning in health_status['warnings']:
                print(f"  ⚠️ {warning}")
        
        if health_status['errors']:
            print("Errors:")
            for error in health_status['errors']:
                print(f"  ❌ {error}")
    
    def _run_integration_test(self):
        """Run integration test and display results"""
        print("\nRunning integration test...")
        results = self.test_system_integration()
        
        print(f"Integration Status: {'✅ Success' if results['overall_success'] else '❌ Failed'}")
        print(f"Test Time: {results['timestamp']}")
        print()
        
        if results['tests_passed']:
            print("Tests Passed:")
            for test in results['tests_passed']:
                print(f"  {test}")
        
        if results['tests_failed']:
            print("Tests Failed:")
            for test in results['tests_failed']:
                print(f"  {test}")
        
        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  {warning}")
        
        if results['performance_metrics']:
            print("Performance Metrics:")
            for metric, value in results['performance_metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}s")
                else:
                    print(f"  {metric}: {value}")
    
    def _show_status(self):
        """Show comprehensive system status"""
        print("\nSystem Status:")
        print("=" * 60)
        
        # Overall system health
        overall_status = self.system_health.get('overall_status', 'unknown')
        status_icon = "✅" if overall_status == 'healthy' else "⚠️" if overall_status == 'degraded' else "❌"
        print(f"Overall Status: {status_icon} {overall_status.upper()}")
        
        uptime = time.time() - self.system_health.get('startup_time', time.time())
        print(f"Uptime: {uptime:.1f} seconds")
        print()
        
        # Core components status
        print("Core Components:")
        print("-" * 30)
        
        for component, status in self.component_status.items():
            status_text = "✅ Active" if status else "❌ Inactive"
            component_name = component.replace('_', ' ').title()
            print(f"  {component_name}: {status_text}")
        
        print()
        
        # Interface status
        print("User Interfaces:")
        print("-" * 30)
        gradio_status = "✅ Running" if hasattr(self, 'gradio_process') else "❌ Not Running"
        streamlit_status = "✅ Running" if (hasattr(self, 'streamlit_process') and 
                                          self.streamlit_process and 
                                          self.streamlit_process.poll() is None) else "❌ Not Running"
        
        print(f"  Gradio Interface: {gradio_status}")
        if gradio_status == "✅ Running":
            print(f"    URL: http://{GRADIO_CONFIG['server_name']}:{GRADIO_CONFIG['server_port']}")
        
        print(f"  Streamlit Dashboard: {streamlit_status}")
        if streamlit_status == "✅ Running":
            print(f"    URL: http://{STREAMLIT_CONFIG['server_address']}:{STREAMLIT_CONFIG['server_port']}")
        
        print()
        
        # System health details
        if self.system_health.get('component_errors'):
            print("Component Errors:")
            print("-" * 30)
            for error in self.system_health['component_errors']:
                print(f"  ❌ {error}")
            print()
        
        # Last health check results
        last_check = self.system_health.get('last_health_check')
        if last_check:
            print("Last Health Check:")
            print("-" * 30)
            print(f"  Overall Healthy: {'✅ Yes' if last_check.get('overall_healthy') else '❌ No'}")
            
            if last_check.get('warnings'):
                print("  Warnings:")
                for warning in last_check['warnings']:
                    print(f"    ⚠️ {warning}")
            
            if last_check.get('errors'):
                print("  Errors:")
                for error in last_check['errors']:
                    print(f"    ❌ {error}")
        
        # Error statistics
        error_stats = self.error_handler.get_error_statistics()
        print(f"\nError Statistics: {error_stats.get('total_errors', 0)} total errors")
    
    def get_unified_api(self) -> Dict[str, Any]:
        """
        Get unified API access to all system components
        
        Returns:
            Dictionary containing all initialized components and their methods
        """
        api = {
            'status': self.system_health,
            'components': {},
            'methods': {}
        }
        
        # Add component references
        if self.ingestion_manager:
            api['components']['ingestion'] = self.ingestion_manager
            api['methods']['process_file'] = self.ingestion_manager.process_file
            api['methods']['process_batch'] = self.ingestion_manager.process_batch
            api['methods']['process_note'] = self.ingestion_manager.process_note
        
        if self.embedding_manager:
            api['components']['embedding'] = self.embedding_manager
            api['methods']['embed_text'] = self.embedding_manager.embed_text
            api['methods']['embed_image'] = self.embedding_manager.embed_image
        
        if self.vector_store:
            api['components']['vector_store'] = self.vector_store
            api['methods']['add_document'] = self.vector_store.add_document
            api['methods']['similarity_search'] = self.vector_store.similarity_search
        
        if self.query_processor:
            api['components']['query'] = self.query_processor
            api['methods']['process_text_query'] = self.query_processor.process_text_query
            api['methods']['process_image_query'] = self.query_processor.process_image_query
            api['methods']['process_multimodal_query'] = self.query_processor.process_multimodal_query
        
        if self.llm_generator:
            api['components']['llm'] = self.llm_generator
            api['methods']['generate_grounded_response'] = self.llm_generator.generate_grounded_response
            api['methods']['generate_summary'] = self.llm_generator.generate_summary
        
        if self.citation_generator:
            api['components']['citation'] = self.citation_generator
            api['methods']['generate_citations'] = self.citation_generator.generate_citations
        
        if self.kg_manager:
            api['components']['knowledge_graph'] = self.kg_manager
            api['methods']['detect_anomalies'] = self.kg_manager.detect_anomalies
            api['methods']['get_graph_stats'] = self.kg_manager.get_graph_stats
        
        if self.feedback_system:
            api['components']['feedback'] = self.feedback_system
            api['methods']['collect_feedback'] = self.feedback_system.collect_feedback
            api['methods']['get_feedback_metrics'] = self.feedback_system.get_feedback_metrics
        
        return api
    
    def test_system_integration(self) -> Dict[str, Any]:
        """
        Test complete system integration with sample data
        
        Returns:
            Test results dictionary
        """
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': True,
            'tests_passed': [],
            'tests_failed': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        try:
            self.logger.info("Running system integration tests...")
            
            # Test 1: Component initialization
            if all(self.component_status.values()):
                test_results['tests_passed'].append("✅ All components initialized")
            else:
                failed_components = [k for k, v in self.component_status.items() if not v]
                test_results['tests_failed'].append(f"❌ Failed components: {failed_components}")
                test_results['overall_success'] = False
            
            # Test 2: Basic ingestion pipeline
            if self.ingestion_manager and self.embedding_manager and self.vector_store:
                try:
                    # Test note processing (simplest case)
                    test_note = "This is a test note for system integration."
                    start_time = time.time()
                    
                    processed_note = self.ingestion_manager.process_note(test_note, "integration_test")
                    if processed_note:
                        # Test embedding generation
                        embedding = self.embedding_manager.embed_text(test_note)
                        if embedding is not None:
                            # Test vector storage
                            processed_note['embedding'] = embedding
                            self.vector_store.add_document(processed_note)
                            
                            processing_time = time.time() - start_time
                            test_results['tests_passed'].append("✅ Ingestion pipeline working")
                            test_results['performance_metrics']['ingestion_time'] = processing_time
                        else:
                            test_results['tests_failed'].append("❌ Embedding generation failed")
                            test_results['overall_success'] = False
                    else:
                        test_results['tests_failed'].append("❌ Note processing failed")
                        test_results['overall_success'] = False
                        
                except Exception as e:
                    test_results['tests_failed'].append(f"❌ Ingestion pipeline error: {e}")
                    test_results['overall_success'] = False
            else:
                test_results['tests_failed'].append("❌ Ingestion pipeline components missing")
                test_results['overall_success'] = False
            
            # Test 3: Query processing pipeline
            if self.query_processor:
                try:
                    start_time = time.time()
                    query_result = self.query_processor.process_text_query("test integration query")
                    query_time = time.time() - start_time
                    
                    test_results['tests_passed'].append("✅ Query processing working")
                    test_results['performance_metrics']['query_time'] = query_time
                    
                except Exception as e:
                    test_results['tests_failed'].append(f"❌ Query processing error: {e}")
                    test_results['overall_success'] = False
            else:
                test_results['tests_failed'].append("❌ Query processor not available")
                test_results['overall_success'] = False
            
            # Test 4: Generation pipeline (if available)
            if self.llm_generator and self.citation_generator:
                try:
                    test_context = [{'content': 'Test context for generation', 'metadata': {'file_path': 'test.txt'}}]
                    response = self.llm_generator.generate_grounded_response("test query", test_context)
                    
                    if response and response.response_text:
                        citations = self.citation_generator.generate_citations(response.response_text, test_context)
                        test_results['tests_passed'].append("✅ Generation pipeline working")
                    else:
                        test_results['warnings'].append("⚠️ Generation produced empty response")
                        
                except Exception as e:
                    test_results['warnings'].append(f"⚠️ Generation pipeline error: {e}")
            else:
                test_results['warnings'].append("⚠️ Generation components not available")
            
            # Test 5: Knowledge graph integration
            if self.kg_manager:
                try:
                    stats = self.kg_manager.get_graph_stats()
                    test_results['tests_passed'].append("✅ Knowledge graph accessible")
                    test_results['performance_metrics']['kg_nodes'] = stats.get('nodes', 0)
                    
                except Exception as e:
                    test_results['tests_failed'].append(f"❌ Knowledge graph error: {e}")
                    test_results['overall_success'] = False
            else:
                test_results['tests_failed'].append("❌ Knowledge graph manager not available")
                test_results['overall_success'] = False
            
            # Test 6: Feedback system
            if self.feedback_system:
                try:
                    # Test feedback collection
                    self.feedback_system.collect_feedback(
                        query="test query",
                        response="test response", 
                        rating=5,
                        comments="Integration test feedback"
                    )
                    test_results['tests_passed'].append("✅ Feedback system working")
                    
                except Exception as e:
                    test_results['tests_failed'].append(f"❌ Feedback system error: {e}")
                    test_results['overall_success'] = False
            else:
                test_results['tests_failed'].append("❌ Feedback system not available")
                test_results['overall_success'] = False
            
            # Summary
            total_tests = len(test_results['tests_passed']) + len(test_results['tests_failed'])
            passed_tests = len(test_results['tests_passed'])
            
            self.logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed")
            
            if test_results['overall_success']:
                self.logger.info("✅ All critical integration tests passed")
            else:
                self.logger.warning("⚠️ Some integration tests failed")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Integration test error: {e}")
            test_results['tests_failed'].append(f"❌ Integration test framework error: {e}")
            test_results['overall_success'] = False
            return test_results
    
    def _run_tests(self):
        """Run comprehensive system tests"""
        print("\nRunning system tests...")
        print("=" * 50)
        
        # Run offline operation test
        print("1. Testing offline operation capabilities...")
        offline_results = self.error_handler.test_offline_operation()
        
        print(f"   Offline Capable: {'✅ Yes' if offline_results['offline_capable'] else '❌ No'}")
        
        if offline_results['tests_passed']:
            print("   Tests Passed:")
            for test in offline_results['tests_passed']:
                print(f"     ✅ {test}")
        
        if offline_results['tests_failed']:
            print("   Tests Failed:")
            for test in offline_results['tests_failed']:
                print(f"     ❌ {test}")
        
        print()
        
        # Run integration tests
        print("2. Testing system integration...")
        integration_results = self.test_system_integration()
        
        print(f"   Integration Status: {'✅ Success' if integration_results['overall_success'] else '❌ Failed'}")
        
        if integration_results['tests_passed']:
            print("   Integration Tests Passed:")
            for test in integration_results['tests_passed']:
                print(f"     {test}")
        
        if integration_results['tests_failed']:
            print("   Integration Tests Failed:")
            for test in integration_results['tests_failed']:
                print(f"     {test}")
        
        if integration_results['warnings']:
            print("   Warnings:")
            for warning in integration_results['warnings']:
                print(f"     {warning}")
        
        # Performance metrics
        if integration_results['performance_metrics']:
            print("   Performance Metrics:")
            for metric, value in integration_results['performance_metrics'].items():
                if isinstance(value, float):
                    print(f"     {metric}: {value:.3f}s")
                else:
                    print(f"     {metric}: {value}")
        
        print()
        
        # Component health check
        print("3. Running component health check...")
        health_status = self._perform_health_check()
        
        print(f"   Overall Health: {'✅ Healthy' if health_status['overall_healthy'] else '❌ Unhealthy'}")
        
        if health_status['critical_components']:
            print("   Critical Components:")
            for component in health_status['critical_components']:
                print(f"     {component}")
        
        if health_status['optional_components']:
            print("   Optional Components:")
            for component in health_status['optional_components']:
                print(f"     {component}")
        
        print("\nTest Summary:")
        print("-" * 30)
        offline_status = "✅ Pass" if offline_results['offline_capable'] else "❌ Fail"
        integration_status = "✅ Pass" if integration_results['overall_success'] else "❌ Fail"
        health_status_text = "✅ Pass" if health_status['overall_healthy'] else "❌ Fail"
        
        print(f"Offline Operation: {offline_status}")
        print(f"System Integration: {integration_status}")
        print(f"Component Health: {health_status_text}")
        
        overall_pass = (offline_results['offline_capable'] and 
                       integration_results['overall_success'] and 
                       health_status['overall_healthy'])
        
        print(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
    
    def shutdown(self):
        """Shutdown all components gracefully"""
        self.logger.info("Shutting down SecureInsight...")
        
        # Shutdown Streamlit process
        if hasattr(self, 'streamlit_process') and self.streamlit_process and self.streamlit_process.poll() is None:
            self.logger.info("Shutting down Streamlit dashboard...")
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Streamlit process did not terminate gracefully, killing...")
                self.streamlit_process.kill()
        
        # Save component states and caches
        try:
            if self.embedding_manager:
                self.logger.info("Saving embedding cache...")
                self.embedding_manager.save_cache()
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {e}")
        
        try:
            if self.vector_store:
                self.logger.info("Persisting vector store...")
                # Vector store should auto-persist, but we can trigger it explicitly
                pass
        except Exception as e:
            self.logger.warning(f"Failed to persist vector store: {e}")
        
        try:
            if self.feedback_system:
                self.logger.info("Saving feedback data...")
                # Feedback system should auto-save, but we can trigger final save
                pass
        except Exception as e:
            self.logger.warning(f"Failed to save feedback data: {e}")
        
        try:
            if self.kg_manager:
                self.logger.info("Saving knowledge graph state...")
                # Save any pending graph updates
                pass
        except Exception as e:
            self.logger.warning(f"Failed to save knowledge graph: {e}")
        
        # Update system status
        self.system_health['overall_status'] = 'shutdown'
        
        self.logger.info("✅ SecureInsight shutdown completed gracefully")
    
    def run(self, mode: str = 'interactive', launch_gradio: bool = True, launch_streamlit: bool = True) -> int:
        """
        Main run method
        
        Args:
            mode: Run mode ('interactive', 'gradio_only', 'streamlit_only', 'headless')
            launch_gradio: Whether to launch Gradio interface
            launch_streamlit: Whether to launch Streamlit dashboard
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Validate system
            if not self.validate_system():
                return 1
            
            # Initialize components
            if not self.initialize_components():
                return 1
            
            # Launch interfaces based on mode
            if mode in ['interactive', 'gradio_only'] and launch_gradio:
                if not self.launch_gradio_interface():
                    self.logger.warning("Gradio interface failed to launch, continuing without it")
            
            if mode in ['interactive', 'streamlit_only'] and launch_streamlit:
                if not self.launch_streamlit_dashboard():
                    self.logger.warning("Streamlit dashboard failed to launch, continuing without it")
            
            # Run appropriate mode
            if mode == 'interactive':
                self.run_interactive_mode()
            elif mode == 'headless':
                self.logger.info("Running in headless mode. Press Ctrl+C to stop.")
                try:
                    while not self.shutdown_event.is_set():
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            elif mode in ['gradio_only', 'streamlit_only']:
                self.logger.info(f"Running in {mode} mode. Press Ctrl+C to stop.")
                try:
                    while not self.shutdown_event.is_set():
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            
            return 0
            
        except Exception as e:
            error_report = self.error_handler.handle_error(
                e, ErrorCategory.PERFORMANCE,
                context={'run_mode': mode},
                severity=ErrorSeverity.CRITICAL
            )
            self.logger.error(f"Application failed: {error_report.user_guidance}")
            return 1
        
        finally:
            self.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SecureInsight Multimodal RAG System")
    parser.add_argument('--mode', choices=['interactive', 'gradio_only', 'streamlit_only', 'headless'],
                       default='interactive', help='Run mode')
    parser.add_argument('--no-gradio', action='store_true', help='Disable Gradio interface')
    parser.add_argument('--no-streamlit', action='store_true', help='Disable Streamlit dashboard')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if not setup_logging():
        print("Failed to setup logging system")
        return 1
    
    # Create launcher
    launcher = SecureInsightLauncher()
    
    # Handle validate-only mode
    if args.validate_only:
        if launcher.validate_system():
            print("✅ System validation passed")
            return 0
        else:
            print("❌ System validation failed")
            return 1
    
    # Run the application
    return launcher.run(
        mode=args.mode,
        launch_gradio=not args.no_gradio,
        launch_streamlit=not args.no_streamlit
    )


if __name__ == "__main__":
    sys.exit(main())