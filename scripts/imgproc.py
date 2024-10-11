#!/usr/bin/env python3

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2  # pylint: disable=no-member
from rembg import remove, new_session

DEFAULT_LINE_THICKNESS = 5


def remove_background(image):
    session = new_session("isnet-anime")
    return remove(image, session=session)


def apply_canny(image, line_thickness):
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((line_thickness, line_thickness), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    _, mask = cv2.threshold(thick_edges, 50, 255, cv2.THRESH_BINARY)
    result = np.zeros_like(img_array)
    result[mask != 0] = [255, 255, 255]
    pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return Image.fromarray(255 - np.array(pil_image))


def save_image(image, path):
    image.save(path)
    print(f"Image saved: {path}")


def process_image(
    image_path,
    line_thickness,
    apply_rembg,
    apply_canny_filter,
):
    with Image.open(image_path) as img:
        if apply_rembg or apply_canny_filter:
            img_no_bg = remove_background(img)
            if apply_rembg:
                yield "rembg", img_no_bg
            if apply_canny_filter:
                canny_image = apply_canny(img_no_bg, line_thickness)
                yield "canny", canny_image


def main():
    parser = argparse.ArgumentParser(
        description="Process image with various techniques."
    )
    parser.add_argument("image", help="Image file to process")
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=DEFAULT_LINE_THICKNESS,
        help=f"Thickness of the lines for Canny edge detection (default: {DEFAULT_LINE_THICKNESS})",
    )
    parser.add_argument(
        "--rembg", action="store_true", help="Apply background removal"
    )
    parser.add_argument(
        "--canny", action="store_true", help="Apply Canny edge detection"
    )

    args = parser.parse_args()

    if not (args.rembg or args.canny):
        parser.error("At least one of --rembg or --canny must be specified")

    input_path = Path(args.image)
    stem = input_path.stem

    for process_type, processed_image in process_image(
        input_path,
        args.line_thickness,
        args.rembg,
        args.canny,
    ):
        new_stem = f"{stem} imgproc-{process_type}"
        output_path = input_path.with_stem(new_stem).with_suffix(".png")
        save_image(processed_image, output_path)


if __name__ == "__main__":
    main()
