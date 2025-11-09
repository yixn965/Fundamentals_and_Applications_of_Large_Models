import os, sys, io
import time
from datetime import datetime

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def setup_stdout_stderr_tee(save_dir, filename=f"log_{current_datetime}.txt"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)
    f = open(log_path, "w", buffering=1, encoding="utf-8")  # line-buffered
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    print(f"[logger] capturing stdout/stderr to {log_path}")
    return f
