import os

# Create a simple global state file or use an environment variable/temp file to communicate status
# A better way is to write the status to a temp file that the dashboard can read
STATUS_FILE = "/tmp/ws_status.txt"

def set_ws_status(connected: bool):
    with open(STATUS_FILE, "w") as f:
        f.write("connected" if connected else "disconnected")

def get_ws_status() -> bool:
    if not os.path.exists(STATUS_FILE):
        return False
    with open(STATUS_FILE, "r") as f:
        return f.read().strip() == "connected"
