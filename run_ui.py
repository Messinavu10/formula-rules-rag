#!/usr/bin/env python3
"""
FIA Formula 1 Regulations Chatbot Launcher
Simple launcher script for the Streamlit UI
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit UI."""
    
    print("🏎️  FIA Formula 1 Regulations Chatbot")
    print("=" * 50)
    print("🚀 Starting Streamlit UI...")
    print("📱 The interface will open in your browser")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit  # type: ignore
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Streamlit: {e}")
            print("💡 Please install manually: pip install streamlit")
            return
    
    # Get the path to the streamlit app
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app not found at {app_path}")
        return
    
    print(f"📱 Launching UI from: {app_path}")
    print("🌐 Opening browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

if __name__ == "__main__":
    main()
