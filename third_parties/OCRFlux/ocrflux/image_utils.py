import os
import subprocess
import io
from typing import List, Union
from PIL import Image


def get_page_image(pdf_path, page_number, target_longest_image_dim=None, image_rotation=0):
    if pdf_path.lower().endswith(".pdf"):
        # Convert PDF page to PNG using pdftoppm
        pdftoppm_result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-r",
                "72",  # 72 pixels per point is the conversion factor
                pdf_path,
            ],
            timeout=120,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
        image = Image.open(io.BytesIO(pdftoppm_result.stdout))
    else:
        image = Image.open(pdf_path)
    if image_rotation != 0:
        image = image.rotate(-image_rotation, expand=True)
    if target_longest_image_dim is not None:
        width, height = image.size
        if width > height:
            new_width = target_longest_image_dim
            new_height = int(height * (target_longest_image_dim / width))
        else:
            new_height = target_longest_image_dim
            new_width = int(width * (target_longest_image_dim / height))
        image = image.resize((new_width, new_height))
    return image


def is_image(file_path):
    try:
        Image.open(file_path)
        return True
    except:
        return False
