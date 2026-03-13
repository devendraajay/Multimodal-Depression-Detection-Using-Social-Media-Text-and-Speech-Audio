"""
Simple script to run the Streamlit app
"""
import subprocess
import sys
import os

if __name__ == '__main__':
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    print(f"Starting Streamlit app: {app_path}")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])