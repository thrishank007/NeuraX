#!/usr/bin/env python3
"""
Automated dependency installation script for SecureInsight RAG system
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
from typing import List, Dict, Optional
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def run_command(command: List[str], description: str) -> bool:
    """Run a command and return success status"""
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_pip_requirements():
    """Install Python packages from requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip first
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing Python dependencies")

def install_system_dependencies():
    """Install system-level dependencies based on OS"""
    system = platform.system().lower()
    
    if system == "linux":
        return install_linux_dependencies()
    elif system == "darwin":  # macOS
        return install_macos_dependencies()
    elif system == "windows":
        return install_windows_dependencies()
    else:
        print(f"⚠️  Unsupported operating system: {system}")
        return True  # Continue anyway

def install_linux_dependencies():
    """Install Linux system dependencies"""
    print("🐧 Detected Linux system")
    
    # Check if running as root or with sudo
    if os.geteuid() != 0:
        print("⚠️  Some system dependencies may require sudo privileges")
    
    # Try to install Tesseract OCR
    commands = [
        (["sudo", "apt-get", "update"], "Updating package list"),
        (["sudo", "apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng"], 
         "Installing Tesseract OCR"),
        (["sudo", "apt-get", "install", "-y", "ffmpeg"], "Installing FFmpeg for audio processing"),
        (["sudo", "apt-get", "install", "-y", "libsndfile1"], "Installing audio libraries")
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  Failed to install system dependency: {description}")
            success = False
    
    return success

def install_macos_dependencies():
    """Install macOS system dependencies"""
    print("🍎 Detected macOS system")
    
    # Check if Homebrew is installed
    try:
        subprocess.run(["brew", "--version"], check=True, capture_output=True)
        print("✅ Homebrew detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Homebrew not found. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    
    commands = [
        (["brew", "install", "tesseract"], "Installing Tesseract OCR"),
        (["brew", "install", "ffmpeg"], "Installing FFmpeg for audio processing")
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success

def install_windows_dependencies():
    """Install Windows system dependencies"""
    print("🪟 Detected Windows system")
    
    print("⚠️  Manual installation required for Windows:")
    print("1. Download and install Tesseract OCR from:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Add Tesseract to your PATH environment variable")
    print("3. Download and install FFmpeg from:")
    print("   https://ffmpeg.org/download.html")
    print("4. Add FFmpeg to your PATH environment variable")
    
    return True

def verify_installations():
    """Verify that key dependencies are properly installed"""
    print("\n🔍 Verifying installations...")
    
    # Check Python packages
    packages_to_check = [
        "torch", "transformers", "sentence_transformers", 
        "chromadb", "gradio", "streamlit", "loguru",
        "numpy", "pandas", "PIL", "whisper"
    ]
    
    missing_packages = []
    for package in packages_to_check:
        if check_package_installed(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    # Check system dependencies
    system_deps = [
        ("tesseract", "Tesseract OCR"),
        ("ffmpeg", "FFmpeg")
    ]
    
    for command, name in system_deps:
        try:
            subprocess.run([command, "--version"], check=True, capture_output=True)
            print(f"✅ {name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("\n✅ All verifications passed!")
    return True

def create_model_directories():
    """Create necessary model directories"""
    from config import MODELS_DIR, DATA_DIR, VECTOR_DB_DIR, LOGS_DIR
    
    directories = [MODELS_DIR, DATA_DIR, VECTOR_DB_DIR, LOGS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_logging():
    """Initialize logging configuration"""
    try:
        from loguru import logger
        from config import LOGGING_CONFIG, LOGS_DIR
        
        # Remove default handler
        logger.remove()
        
        # Add file handler
        log_file = LOGS_DIR / "secureinsight.log"
        logger.add(
            log_file,
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"],
            rotation=LOGGING_CONFIG["rotation"],
            retention=LOGGING_CONFIG.get("retention", "1 week"),
            compression=LOGGING_CONFIG.get("compression", "gz")
        )
        
        # Add console handler
        logger.add(
            sys.stderr,
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"]
        )
        
        logger.info("Logging system initialized")
        print("✅ Logging system configured")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup logging: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 SecureInsight RAG System - Dependency Installation")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_model_directories()
    
    # Install system dependencies
    print("\n🔧 Installing system dependencies...")
    install_system_dependencies()
    
    # Install Python packages
    print("\n🐍 Installing Python packages...")
    if not install_pip_requirements():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Setup logging
    print("\n📝 Setting up logging...")
    setup_logging()
    
    # Verify installations
    if not verify_installations():
        print("\n⚠️  Some dependencies may not be properly installed")
        print("Please check the error messages above and install missing dependencies manually")
    
    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Run 'python download_models.py' to download offline models")
    print("2. Run 'python main_launcher.py' to start the system")

if __name__ == "__main__":
    main()