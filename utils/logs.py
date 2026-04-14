from datetime import datetime
from pathlib import Path


def log(message: str, log_file: Path = None):
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
