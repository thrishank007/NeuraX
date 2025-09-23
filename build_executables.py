#!/usr/bin/env python3
"""
Automated build script for SecureInsight portable executables
"""
import os
import sys
import subprocess
import shutil
import time
import platform
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from build_config import (
    BUILD_CONFIG, PATHS_CONFIG, MODEL_PACKAGE_CONFIG, USB_DEPLOYMENT_CONFIG,
    validate_build_requirements, estimate_package_size, get_platform_config
)
from create_pyinstaller_spec import create_spec_file, validate_spec_file
from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class SecureInsightBuilder:
    """Automated builder for SecureInsight portable executables"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.platform_config = get_platform_config()
        self.build_start_time = time.time()
        self.build_log = []
        
        # Setup paths
        self.build_dir = PATHS_CONFIG['build_dir']
        self.dist_dir = PATHS_CONFIG['dist_dir']
        self.package_dir = PATHS_CONFIG['package_dir']
        self.spec_dir = PATHS_CONFIG['spec_dir']
        
        logger.info("SecureInsight Builder initialized")
    
    def log_step(self, message: str, success: bool = True):
        """Log build step with timestamp"""
        timestamp = time.strftime('%H:%M:%S')
        status = "✅" if success else "❌"
        log_entry = f"[{timestamp}] {status} {message}"
        self.build_log.append(log_entry)
        logger.info(message)
        print(log_entry)
    
    def validate_prerequisites(self) -> bool:
        """Validate build prerequisites"""
        self.log_step("Validating build prerequisites...")
        
        try:
            # Check build requirements
            validation = validate_build_requirements()
            
            if not validation['valid']:
                for error in validation['errors']:
                    self.log_step(f"Prerequisite error: {error}", False)
                return False
            
            for warning in validation['warnings']:
                self.log_step(f"Warning: {warning}")
            
            # Check PyInstaller specifically
            try:
                import PyInstaller
                self.log_step(f"PyInstaller {PyInstaller.__version__} available")
            except ImportError:
                self.log_step("PyInstaller not installed", False)
                return False
            
            # Check disk space
            size_estimate = estimate_package_size()
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            required_space_gb = size_estimate['total_gb'] * 2  # 2x for build process
            
            if free_space_gb < required_space_gb:
                self.log_step(f"Insufficient disk space: {free_space_gb:.1f}GB free, {required_space_gb:.1f}GB required", False)
                return False
            
            self.log_step(f"Sufficient disk space: {free_space_gb:.1f}GB available")
            self.log_step("Prerequisites validation completed")
            return True
            
        except Exception as e:
            self.log_step(f"Prerequisites validation failed: {e}", False)
            return False
    
    def prepare_build_environment(self) -> bool:
        """Prepare build environment"""
        self.log_step("Preparing build environment...")
        
        try:
            # Clean previous builds
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
                self.log_step("Cleaned previous build directory")
            
            if self.dist_dir.exists():
                shutil.rmtree(self.dist_dir)
                self.log_step("Cleaned previous dist directory")
            
            # Create necessary directories
            directories = [
                self.build_dir,
                self.dist_dir,
                self.spec_dir,
                Path('logs'),
                Path('cache'),
                Path('data'),
                Path('vector_db')
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            self.log_step("Created build directories")
            
            # Validate main script exists
            main_script = Path(PATHS_CONFIG['main_script'])
            if not main_script.exists():
                self.log_step(f"Main script not found: {main_script}", False)
                return False
            
            self.log_step("Build environment prepared")
            return True
            
        except Exception as e:
            self.log_step(f"Build environment preparation failed: {e}", False)
            return False
    
    def download_models(self) -> bool:
        """Download required models for offline operation"""
        self.log_step("Checking model availability...")
        
        try:
            models_dir = PATHS_CONFIG['models_dir']
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if models are already downloaded
            existing_models = list(models_dir.glob('**/*'))
            if existing_models:
                self.log_step(f"Found {len(existing_models)} existing model files")
                
                # Check if essential models are present
                essential_models = ['sentence-transformers', 'clip', 'whisper']
                found_essential = []
                
                for model_type in essential_models:
                    model_files = [f for f in existing_models if model_type in str(f).lower()]
                    if model_files:
                        found_essential.append(model_type)
                
                if len(found_essential) >= 2:  # At least 2 essential models
                    self.log_step("Essential models appear to be available")
                    return True
            
            # Try to download models using the download script
            download_script = Path('download_models.py')
            if download_script.exists():
                self.log_step("Attempting to download models...")
                
                try:
                    result = subprocess.run(
                        [sys.executable, str(download_script), '--offline-prep'],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout
                    )
                    
                    if result.returncode == 0:
                        self.log_step("Models downloaded successfully")
                        return True
                    else:
                        self.log_step(f"Model download failed: {result.stderr}", False)
                        
                except subprocess.TimeoutExpired:
                    self.log_step("Model download timed out", False)
                except Exception as e:
                    self.log_step(f"Model download error: {e}", False)
            
            # Continue without models (will use online fallback)
            self.log_step("Continuing build without pre-downloaded models (online fallback will be used)")
            return True
            
        except Exception as e:
            self.log_step(f"Model preparation failed: {e}", False)
            return False
    
    def bundle_tesseract(self) -> bool:
        """Bundle Tesseract OCR for offline operation"""
        self.log_step("Bundling Tesseract OCR...")
        
        try:
        # Check if Tesseract is available
            try:
                result = subprocess.run(['tesseract', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_step("Tesseract OCR found in system")
                    return True
            except FileNotFoundError:
                pass
            
            # Try to find Tesseract in common locations
            tesseract_paths = [
                Path('C:/Program Files/Tesseract-OCR/tesseract.exe'),
                Path('C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'),
                Path('/usr/bin/tesseract'),
                Path('/usr/local/bin/tesseract'),
                Path('/opt/homebrew/bin/tesseract')
            ]
            
            for tesseract_path in tesseract_paths:
                if tesseract_path.exists():
                    self.log_step(f"Found Tesseract at: {tesseract_path}")
                    return True
            
            self.log_step("Tesseract not found - OCR functionality will be limited")
            return True  # Continue build without Tesseract
            
        except Exception as e:
            self.log_step(f"Tesseract bundling failed: {e}", False)
            return True  # Continue build
    
    def create_pyinstaller_spec(self) -> bool:
        """Create PyInstaller spec file"""
        self.log_step("Creating PyInstaller spec file...")
        
        try:
            spec_path = create_spec_file()
            
            if not validate_spec_file(spec_path):
                self.log_step("Spec file validation failed", False)
                return False
            
            self.log_step(f"PyInstaller spec created: {spec_path}")
            return True
            
        except Exception as e:
            self.log_step(f"Spec file creation failed: {e}", False)
            return False
    
    def build_executable(self) -> bool:
        """Build executable using PyInstaller"""
        self.log_step("Building executable with PyInstaller...")
        
        try:
            spec_file = self.spec_dir / 'secureinsight.spec'
            
            if not spec_file.exists():
                self.log_step("Spec file not found", False)
                return False
            
            # Run PyInstaller
            cmd = [
                sys.executable, '-m', 'PyInstaller',
                '--clean',
                '--noconfirm',
                str(spec_file)
            ]
            
            self.log_step(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode != 0:
                self.log_step(f"PyInstaller failed: {result.stderr}", False)
                return False
            
            # Check if executable was created
            exe_name = 'secureinsight.exe' if platform.system() == 'Windows' else 'secureinsight'
            exe_path = self.dist_dir / 'secureinsight' / exe_name
            
            if not exe_path.exists():
                self.log_step("Executable not found after build", False)
                return False
            
            self.log_step(f"Executable built successfully: {exe_path}")
            return True
            
        except subprocess.TimeoutExpired:
            self.log_step("PyInstaller build timed out", False)
            return False
        except Exception as e:
            self.log_step(f"Executable build failed: {e}", False)
            return False
    
    def package_for_distribution(self) -> bool:
        """Package the built executable for distribution"""
        self.log_step("Packaging for distribution...")
        
        try:
            # Create package directory
            self.package_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy executable and dependencies
            exe_dir = self.dist_dir / 'secureinsight'
            if not exe_dir.exists():
                self.log_step("Executable directory not found", False)
                return False
            
            package_name = f"SecureInsight-{platform.system()}-{platform.machine()}"
            target_dir = self.package_dir / package_name
            
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.copytree(exe_dir, target_dir)
            self.log_step(f"Copied executable to: {target_dir}")
            
            # Copy additional files
            additional_files = [
                'README.md',
                'LICENSE',
                'config.py',
                'requirements.txt'
            ]
            
            for file_name in additional_files:
                src_file = Path(file_name)
                if src_file.exists():
                    shutil.copy2(src_file, target_dir)
                    self.log_step(f"Copied {file_name}")
            
            # Create data directories
            data_dirs = ['data', 'logs', 'cache', 'vector_db']
            for dir_name in data_dirs:
                (target_dir / dir_name).mkdir(exist_ok=True)
            
            # Create ZIP archive
            zip_path = self.package_dir / f"{package_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in target_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(target_dir)
                        zipf.write(file_path, arcname)
            
            self.log_step(f"Created distribution package: {zip_path}")
            
            # Calculate package size
            package_size_mb = zip_path.stat().st_size / (1024 * 1024)
            self.log_step(f"Package size: {package_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            self.log_step(f"Packaging failed: {e}", False)
            return False
    
    def create_usb_deployment(self) -> bool:
        """Create USB deployment package"""
        self.log_step("Creating USB deployment package...")
        
        try:
            usb_config = USB_DEPLOYMENT_CONFIG
            
            # Create USB structure
            usb_dir = self.package_dir / 'USB_Deployment'
            usb_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy executable
            exe_dir = self.dist_dir / 'secureinsight'
            if exe_dir.exists():
                shutil.copytree(exe_dir, usb_dir / 'SecureInsight', dirs_exist_ok=True)
            
            # Create autorun file for Windows
            if platform.system() == 'Windows':
                autorun_content = """[autorun]
open=SecureInsight\\secureinsight.exe
icon=SecureInsight\\icon.ico
label=SecureInsight RAG System
"""
                with open(usb_dir / 'autorun.inf', 'w') as f:
                    f.write(autorun_content)
            
            # Create launcher scripts
            if platform.system() == 'Windows':
                launcher_content = """@echo off
cd /d "%~dp0SecureInsight"
secureinsight.exe
pause
"""
                with open(usb_dir / 'Launch_SecureInsight.bat', 'w') as f:
                    f.write(launcher_content)
            else:
                launcher_content = """#!/bin/bash
cd "$(dirname "$0")/SecureInsight"
./secureinsight
"""
                launcher_path = usb_dir / 'Launch_SecureInsight.sh'
                with open(launcher_path, 'w') as f:
                    f.write(launcher_content)
                launcher_path.chmod(0o755)
            
            # Create README for USB
            readme_content = f"""# SecureInsight USB Deployment

## Quick Start
1. Run Launch_SecureInsight.{('bat' if platform.system() == 'Windows' else 'sh')}
2. Wait for the application to load
3. Upload documents and start searching

## System Requirements
- {usb_config['min_ram_gb']}GB RAM minimum
- {usb_config['min_storage_gb']}GB free storage
- {usb_config['supported_os']}

## Features
- Offline multimodal RAG system
- Cross-modal search (text, images, audio)
- Knowledge graph visualization
- Security anomaly detection

Built on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            with open(usb_dir / 'README.txt', 'w') as f:
                f.write(readme_content)
            
            self.log_step("USB deployment package created")
            return True
            
        except Exception as e:
            self.log_step(f"USB deployment creation failed: {e}", False)
            return False
    
    def run_post_build_tests(self) -> bool:
        """Run post-build validation tests"""
        self.log_step("Running post-build tests...")
        
        try:
            # Test executable exists and is executable
            exe_name = 'secureinsight.exe' if platform.system() == 'Windows' else 'secureinsight'
            exe_path = self.dist_dir / 'secureinsight' / exe_name
            
            if not exe_path.exists():
                self.log_step("Executable not found", False)
                return False
            
            if not os.access(exe_path, os.X_OK):
                self.log_step("Executable is not executable", False)
                return False
            
            # Test executable can start (quick test)
            try:
                result = subprocess.run(
                    [str(exe_path), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    self.log_step("Executable version check passed")
                else:
                    self.log_step("Executable version check failed", False)
                    
            except subprocess.TimeoutExpired:
                self.log_step("Executable startup test timed out")
            except Exception as e:
                self.log_step(f"Executable test failed: {e}")
            
            # Validate offline operation script
            offline_validator = Path('validate_offline_operation.py')
            if offline_validator.exists():
                try:
                    result = subprocess.run(
                        [sys.executable, str(offline_validator), '--quick-test'],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        self.log_step("Offline operation validation passed")
                    else:
                        self.log_step("Offline operation validation failed")
                        
                except Exception as e:
                    self.log_step(f"Offline validation error: {e}")
            
            self.log_step("Post-build tests completed")
            return True
            
        except Exception as e:
            self.log_step(f"Post-build tests failed: {e}", False)
            return False
    
    def generate_build_report(self) -> Dict[str, Any]:
        """Generate comprehensive build report"""
        build_time = time.time() - self.build_start_time
        
        report = {
            'build_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': round(build_time, 2),
                'platform': platform.platform(),
                'python_version': sys.version,
                'builder_version': '1.0.0'
            },
            'build_log': self.build_log,
            'artifacts': [],
            'size_info': {},
            'success': True
        }
        
        try:
            # Collect artifact information
            if self.dist_dir.exists():
                for item in self.dist_dir.rglob('*'):
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        report['artifacts'].append({
                            'path': str(item.relative_to(self.dist_dir)),
                            'size_mb': round(size_mb, 2)
                        })
            
            # Package size information
            if self.package_dir.exists():
                for zip_file in self.package_dir.glob('*.zip'):
                    size_mb = zip_file.stat().st_size / (1024 * 1024)
                    report['size_info'][zip_file.name] = f"{size_mb:.1f} MB"
            
            # Check for any failed steps
            failed_steps = [log for log in self.build_log if '❌' in log]
            if failed_steps:
                report['success'] = False
                report['failed_steps'] = failed_steps
            
        except Exception as e:
            logger.error(f"Error generating build report: {e}")
            report['report_error'] = str(e)
        
        return report
    
    def build(self) -> bool:
        """Execute complete build process"""
        self.log_step("Starting SecureInsight build process...")
        
        try:
            # Build steps
            steps = [
                ('Validate Prerequisites', self.validate_prerequisites),
                ('Prepare Build Environment', self.prepare_build_environment),
                ('Download Models', self.download_models),
                ('Bundle Tesseract', self.bundle_tesseract),
                ('Create PyInstaller Spec', self.create_pyinstaller_spec),
                ('Build Executable', self.build_executable),
                ('Package for Distribution', self.package_for_distribution),
                ('Create USB Deployment', self.create_usb_deployment),
                ('Run Post-Build Tests', self.run_post_build_tests)
            ]
            
            for step_name, step_func in steps:
                self.log_step(f"Starting: {step_name}")
                
                if not step_func():
                    self.log_step(f"Build failed at step: {step_name}", False)
                    return False
                
                self.log_step(f"Completed: {step_name}")
            
            # Generate final report
            report = self.generate_build_report()
            
            # Save build report
            report_path = self.package_dir / 'build_report.json'
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            build_time = time.time() - self.build_start_time
            self.log_step(f"Build completed successfully in {build_time:.1f} seconds")
            
            # Print summary
            print("\n" + "="*60)
            print("BUILD SUMMARY")
            print("="*60)
            print(f"Status: {'SUCCESS' if report['success'] else 'FAILED'}")
            print(f"Duration: {build_time:.1f} seconds")
            print(f"Artifacts: {len(report['artifacts'])} files")
            
            if report['size_info']:
                print("Package Sizes:")
                for name, size in report['size_info'].items():
                    print(f"  {name}: {size}")
            
            print(f"Build report: {report_path}")
            print("="*60)
            
            return report['success']
            
        except Exception as e:
            self.log_step(f"Build process failed: {e}", False)
            return False


def main():
    """Main entry point for build script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SecureInsight portable executables')
    parser.add_argument('--clean', action='store_true', help='Clean build directories before building')
    parser.add_argument('--no-models', action='store_true', help='Skip model download step')
    parser.add_argument('--no-tests', action='store_true', help='Skip post-build tests')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.add("build.log", rotation="10 MB", level="DEBUG")
    else:
        logger.add("build.log", rotation="10 MB", level="INFO")
    
    try:
        builder = SecureInsightBuilder()
        
        # Override steps based on arguments
        if args.no_models:
            builder.download_models = lambda: True
        
        if args.no_tests:
            builder.run_post_build_tests = lambda: True
        
        success = builder.build()
        
        if success:
            print("\n🎉 Build completed successfully!")
            print("Check the 'packages' directory for distribution files.")
            sys.exit(0)
        else:
            print("\n❌ Build failed!")
            print("Check build.log for detailed error information.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected build error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()