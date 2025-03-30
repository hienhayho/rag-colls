import os

DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ["true", "1"]
