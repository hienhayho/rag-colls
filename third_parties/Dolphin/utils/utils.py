"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import copy
import io
import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pymupdf
from PIL import Image

from utils.markdown_utils import MarkdownConverter


def save_figure_to_local(pil_crop, save_dir, image_name, reading_order):
    """Save cropped figure to local file system

    Args:
        pil_crop: PIL Image object of the cropped figure
        save_dir: Base directory to save results
        image_name: Name of the source image/document
        reading_order: Reading order of the figure in the document

    Returns:
        str: Filename of the saved figure
    """
    try:
        # Create figures directory if it doesn't exist
        figures_dir = os.path.join(save_dir, "markdown", "figures")
        # os.makedirs(figures_dir, exist_ok=True)

        # Generate figure filename
        figure_filename = f"{image_name}_figure_{reading_order:03d}.png"
        figure_path = os.path.join(figures_dir, figure_filename)

        # Save the figure
        pil_crop.save(figure_path, format="PNG", quality=95)

        # print(f"Saved figure: {figure_filename}")
        return figure_filename

    except Exception as e:
        print(f"Error saving figure: {str(e)}")
        # Return a fallback filename
        return f"{image_name}_figure_{reading_order:03d}_error.png"


def convert_pdf_to_images(pdf_path, target_size=896):
    """Convert PDF pages to images

    Args:
        pdf_path: Path to PDF file
        target_size: Target size for the longest dimension

    Returns:
        List of PIL Images
    """
    images = []
    try:
        doc = pymupdf.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Calculate scale to make longest dimension equal to target_size
            rect = page.rect
            scale = target_size / max(rect.width, rect.height)

            # Render page as image
            mat = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            images.append(pil_image)

        doc.close()
        print(f"Successfully converted {len(images)} pages from PDF")
        return images

    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []


def is_pdf_file(file_path):
    """Check if file is a PDF"""
    return file_path.lower().endswith(".pdf")


def save_combined_pdf_results(all_page_results, pdf_path, save_dir):
    """Save combined results for multi-page PDF with both JSON and Markdown

    Args:
        all_page_results: List of results for all pages
        pdf_path: Path to original PDF file
        save_dir: Directory to save results

    Returns:
        Path to saved combined JSON file
    """
    # Create output filename based on PDF name
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Prepare combined results
    combined_results = {
        "source_file": pdf_path,
        "total_pages": len(all_page_results),
        "pages": all_page_results,
    }

    # Save combined JSON results
    json_filename = f"{base_name}.json"
    json_path = os.path.join(save_dir, "recognition_json", json_filename)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    # Generate and save combined markdown
    try:
        markdown_converter = MarkdownConverter()

        # Combine all page results into a single list for markdown conversion
        all_elements = []
        for page_data in all_page_results:
            page_elements = page_data.get("elements", [])
            if page_elements:
                # Add page separator if not the first page
                if all_elements:
                    all_elements.append(
                        {
                            "label": "page_separator",
                            "text": "\n\n---\n\n",
                            "reading_order": len(all_elements),
                        }
                    )
                all_elements.extend(page_elements)

        # Generate markdown content
        markdown_content = markdown_converter.convert(all_elements)

        # Save markdown file
        markdown_filename = f"{base_name}.md"
        markdown_path = os.path.join(save_dir, "markdown", markdown_filename)
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # print(f"Combined markdown saved to: {markdown_path}")

    except ImportError:
        print("MarkdownConverter not available, skipping markdown generation")
    except Exception as e:
        print(f"Error generating markdown: {e}")

    # print(f"Combined JSON results saved to: {json_path}")
    return json_path


def check_coord_valid(x1, y1, x2, y2, image_size=None, abs_coord=True):
    # print(f"check_coord_valid: {x1}, {y1}, {x2}, {y2}, {image_size}, {abs_coord}")
    if x2 <= x1 or y2 <= y1:
        return False, f"[{x1}, {y1}, {x2}, {y2}]"
    if x1 < 0 or y1 < 0:
        return False, f"[{x1}, {y1}, {x2}, {y2}]"
    if not abs_coord:
        if x2 > 1 or y2 > 1:
            return False, f"[{x1}, {y1}, {x2}, {y2}]"
    elif image_size is not None:  # has image size
        if x2 > image_size[0] or y2 > image_size[1]:
            return False, f"[{x1}, {y1}, {x2}, {y2}]"
    return True, None


def adjust_box_edges(image, boxes: List[List[float]], max_pixels=15, threshold=0.2):
    """
    Image: cv2.image object, or Path
    Input: boxes: list of boxes [[x1, y1, x2, y2]]. Using absolute coordinates.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    img_h, img_w = image.shape[:2]
    new_boxes = []
    for box in boxes:
        best_box = copy.deepcopy(box)

        def check_edge(img, current_box, i, is_vertical):
            edge = current_box[i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            if is_vertical:
                line = binary[current_box[1] : current_box[3] + 1, edge]
            else:
                line = binary[edge, current_box[0] : current_box[2] + 1]

            transitions = np.abs(np.diff(line))
            return np.sum(transitions) / len(transitions)

        # Only widen the box
        edges = [(0, -1, True), (2, 1, True), (1, -1, False), (3, 1, False)]

        current_box = copy.deepcopy(box)
        # make sure the box is within the image
        current_box[0] = min(max(current_box[0], 0), img_w - 1)
        current_box[1] = min(max(current_box[1], 0), img_h - 1)
        current_box[2] = min(max(current_box[2], 0), img_w - 1)
        current_box[3] = min(max(current_box[3], 0), img_h - 1)

        for i, direction, is_vertical in edges:
            best_score = check_edge(image, current_box, i, is_vertical)
            if best_score <= threshold:
                continue
            for step in range(max_pixels):
                current_box[i] += direction
                if i == 0 or i == 2:
                    current_box[i] = min(max(current_box[i], 0), img_w - 1)
                else:
                    current_box[i] = min(max(current_box[i], 0), img_h - 1)
                score = check_edge(image, current_box, i, is_vertical)

                if score < best_score:
                    best_score = score
                    best_box = copy.deepcopy(current_box)

                if score <= threshold:
                    break
        new_boxes.append(best_box)

    return new_boxes


def parse_layout_string(bbox_str):
    """Parse layout string using regular expressions"""
    pattern = r"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)"
    matches = re.finditer(pattern, bbox_str)

    parsed_results = []
    for match in matches:
        coords = [float(match.group(i)) for i in range(1, 5)]
        label = match.group(5).strip()
        parsed_results.append((coords, label))

    return parsed_results


@dataclass
class ImageDimensions:
    """Class to store image dimensions"""

    original_w: int
    original_h: int
    padded_w: int
    padded_h: int


def map_to_original_coordinates(
    x1, y1, x2, y2, dims: ImageDimensions
) -> Tuple[int, int, int, int]:
    """Map coordinates from padded image back to original image

    Args:
        x1, y1, x2, y2: Coordinates in padded image
        dims: Image dimensions object

    Returns:
        tuple: (x1, y1, x2, y2) coordinates in original image
    """
    try:
        # Calculate padding offsets
        top = (dims.padded_h - dims.original_h) // 2
        left = (dims.padded_w - dims.original_w) // 2

        # Map back to original coordinates
        orig_x1 = max(0, x1 - left)
        orig_y1 = max(0, y1 - top)
        orig_x2 = min(dims.original_w, x2 - left)
        orig_y2 = min(dims.original_h, y2 - top)

        # Ensure we have a valid box (width and height > 0)
        if orig_x2 <= orig_x1:
            orig_x2 = min(orig_x1 + 1, dims.original_w)
        if orig_y2 <= orig_y1:
            orig_y2 = min(orig_y1 + 1, dims.original_h)

        return int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)
    except Exception as e:
        print(f"map_to_original_coordinates error: {str(e)}")
        # Return safe coordinates
        return 0, 0, min(100, dims.original_w), min(100, dims.original_h)


def map_to_relevant_coordinates(abs_coords, dims: ImageDimensions):
    """
    From absolute coordinates to relevant coordinates
    e.g. [100, 100, 200, 200] -> [0.1, 0.2, 0.3, 0.4]
    """
    try:
        x1, y1, x2, y2 = abs_coords
        return (
            round(x1 / dims.original_w, 3),
            round(y1 / dims.original_h, 3),
            round(x2 / dims.original_w, 3),
            round(y2 / dims.original_h, 3),
        )
    except Exception as e:
        print(f"map_to_relevant_coordinates error: {str(e)}")
        return 0.0, 0.0, 1.0, 1.0  # Return full image coordinates


def process_coordinates(coords, padded_image, dims: ImageDimensions, previous_box=None):
    """Process and adjust coordinates

    Args:
        coords: Normalized coordinates [x1, y1, x2, y2]
        padded_image: Padded image
        dims: Image dimensions object
        previous_box: Previous box coordinates for overlap adjustment

    Returns:
        tuple: (x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, new_previous_box)
    """
    try:
        # Convert normalized coordinates to absolute coordinates
        x1, y1 = int(coords[0] * dims.padded_w), int(coords[1] * dims.padded_h)
        x2, y2 = int(coords[2] * dims.padded_w), int(coords[3] * dims.padded_h)

        # Ensure coordinates are within image bounds before adjustment
        x1 = max(0, min(x1, dims.padded_w - 1))
        y1 = max(0, min(y1, dims.padded_h - 1))
        x2 = max(0, min(x2, dims.padded_w))
        y2 = max(0, min(y2, dims.padded_h))

        # Ensure width and height are at least 1 pixel
        if x2 <= x1:
            x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1:
            y2 = min(y1 + 1, dims.padded_h)

        # Extend box boundaries
        new_boxes = adjust_box_edges(padded_image, [[x1, y1, x2, y2]])
        x1, y1, x2, y2 = new_boxes[0]

        # Ensure coordinates are still within image bounds after adjustment
        x1 = max(0, min(x1, dims.padded_w - 1))
        y1 = max(0, min(y1, dims.padded_h - 1))
        x2 = max(0, min(x2, dims.padded_w))
        y2 = max(0, min(y2, dims.padded_h))

        # Ensure width and height are at least 1 pixel after adjustment
        if x2 <= x1:
            x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1:
            y2 = min(y1 + 1, dims.padded_h)

        # Check for overlap with previous box and adjust
        if previous_box is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
            if (x1 < prev_x2 and x2 > prev_x1) and (y1 < prev_y2 and y2 > prev_y1):
                y1 = prev_y2
                # Ensure y1 is still valid
                y1 = min(y1, dims.padded_h - 1)
                # Make sure y2 is still greater than y1
                if y2 <= y1:
                    y2 = min(y1 + 1, dims.padded_h)

        # Update previous box
        new_previous_box = [x1, y1, x2, y2]

        # Map to original coordinates
        orig_x1, orig_y1, orig_x2, orig_y2 = map_to_original_coordinates(
            x1, y1, x2, y2, dims
        )

        return x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, new_previous_box
    except Exception as e:
        print(f"process_coordinates error: {str(e)}")
        # Return safe values
        orig_x1, orig_y1, orig_x2, orig_y2 = (
            0,
            0,
            min(100, dims.original_w),
            min(100, dims.original_h),
        )
        return 0, 0, 100, 100, orig_x1, orig_y1, orig_x2, orig_y2, [0, 0, 100, 100]


def prepare_image(image) -> Tuple[np.ndarray, ImageDimensions]:
    """Load and prepare image with padding while maintaining aspect ratio

    Args:
        image: PIL image

    Returns:
        tuple: (padded_image, image_dimensions)
    """
    try:
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_h, original_w = image.shape[:2]

        # Calculate padding to make square image
        max_size = max(original_h, original_w)
        top = (max_size - original_h) // 2
        bottom = max_size - original_h - top
        left = (max_size - original_w) // 2
        right = max_size - original_w - left

        # Apply padding
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        padded_h, padded_w = padded_image.shape[:2]

        dimensions = ImageDimensions(
            original_w=original_w,
            original_h=original_h,
            padded_w=padded_w,
            padded_h=padded_h,
        )

        return padded_image, dimensions
    except Exception as e:
        print(f"prepare_image error: {str(e)}")
        # Create a minimal valid image and dimensions
        h, w = image.height, image.width
        dimensions = ImageDimensions(original_w=w, original_h=h, padded_w=w, padded_h=h)
        # Return a black image of the same size
        return np.zeros((h, w, 3), dtype=np.uint8), dimensions


def setup_output_dirs(save_dir):
    """Create necessary output directories"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "markdown"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "recognition_json"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "markdown", "figures"), exist_ok=True)


def save_outputs(recognition_results, image_path, save_dir):
    """Save JSON and markdown outputs"""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Save JSON file
    json_path = os.path.join(save_dir, "recognition_json", f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recognition_results, f, ensure_ascii=False, indent=2)

    # Generate and save markdown file
    markdown_converter = MarkdownConverter()
    markdown_content = markdown_converter.convert(recognition_results)
    markdown_path = os.path.join(save_dir, "markdown", f"{basename}.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return json_path


def crop_margin(img: Image.Image) -> Image.Image:
    """Crop margins from image"""
    try:
        width, height = img.size
        if width == 0 or height == 0:
            print("Warning: Image has zero width or height")
            return img

        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        if coords is None:
            return img
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

        # Ensure crop coordinates are within image bounds
        a = max(0, a)
        b = max(0, b)
        w = min(w, width - a)
        h = min(h, height - b)

        # Only crop if we have a valid region
        if w > 0 and h > 0:
            return img.crop((a, b, a + w, b + h))
        return img
    except Exception as e:
        print(f"crop_margin error: {str(e)}")
        return img  # Return original image on error
