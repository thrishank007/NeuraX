#!/usr/bin/env python3
"""
Launch script for NeuraX Streamlit Dashboard

This script launches the Streamlit monitoring dashboard for NeuraX.
Run with: python launch_dashboard.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    # Get the path to the dashboard script
    dashboard_path = Path(__file__).parent / "ui" / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard script not found at {dashboard_path}")
        sys.exit(1)
    
    print("🚀 Launching NeuraX Monitoring Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()