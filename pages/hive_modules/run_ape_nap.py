# run_ape_nap.py
import streamlit as st
import subprocess
import sys
import os

# Auto-configure for nap mode
os.environ["HIVE_WATCHDOG_TIMER"] = "3600"
os.environ["MAX_TOOL_CALLS_PER_RUN"] = "500"
os.environ["MAX_COST_PER_RUN"] = "2.0"  # Adjust as needed

# Launch with flags to minimize UI overhead
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "main.py",
    "--server.headless", "true",
    "--server.runOnSave", "false",
    "--theme.base", "dark"
])
