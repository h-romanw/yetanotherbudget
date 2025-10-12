#!/usr/bin/env python3
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0"
    ])
