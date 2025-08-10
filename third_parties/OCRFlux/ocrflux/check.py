import importlib.util
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def check_poppler_version():
    try:
        result = subprocess.run(["pdftoppm", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stderr.startswith("pdftoppm"):
            logger.info("pdftoppm is installed and working.")
        else:
            logger.error("pdftoppm is installed but returned an error.")
            sys.exit(1)
    except FileNotFoundError:
        logger.error("pdftoppm is not installed.")
        sys.exit(1)

def check_vllm_version():
    if importlib.util.find_spec("vllm") is None:
        logger.error("VLLM needs to be installed with a separate command in order to find all dependencies properly.")
        sys.exit(1)


def check_torch_gpu_available(min_gpu_memory: int = 20 * 1024**3):
    try:
        import torch
    except:
        logger.error("Pytorch must be installed, visit https://pytorch.org/ for installation instructions")
        raise

if __name__ == "__main__":
    check_poppler_version()
    check_vllm_version()
