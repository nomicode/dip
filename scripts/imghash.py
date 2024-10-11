#!/usr/bin/env python3

# IGNORE THESE LINES, JUST FOR MY REFERENCES
# imghash --face --facial-features --features-thickness 5 --crop-padding 200
#  --blur 2.5
# imghash --outline --features-thickness 5 --debug --blur 2.5

# pylint: disable=no-member, import-error

import argparse
import shutil
from pathlib import Path

import cv2
import imagehash
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove

# Set a more conservative maximum file name length
MAX_FILE_NAME_LENGTH = 200
DEFAULT_BLUR = 2
DEFAULT_CROP_PADDING = 0
DEFAULT_POSE_LINE_THICKNESS = 8
DEFAULT_POSE_CIRCLE_RADIUS = 4
DEFAULT_FEATURES_THICKNESS = 2


def invert_mask(mask):
    return Image.fromarray(255 - np.array(mask))


def detect_face(image, crop_padding):
    # Convert PIL Image to OpenCV format
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        img_array, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # Get the first face found
        x, y, w, h = faces[0]

        # Add padding
        x = max(0, x - crop_padding)
        y = max(0, y - crop_padding)
        w = min(img_array.shape[1] - x, w + 2 * crop_padding)
        h = min(img_array.shape[0] - y, h + 2 * crop_padding)

        # Crop the face
        face = img_array[y : y + h, x : x + w]

        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    else:
        return image  # Return original image if no face detected


def extract_features(image, line_thickness):
    # Convert PIL Image to OpenCV format
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Dilate the edges to make them thicker
    kernel = np.ones((line_thickness, line_thickness), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a binary mask from the edges
    _, mask = cv2.threshold(thick_edges, 50, 255, cv2.THRESH_BINARY)

    # Create a blank image and apply the mask
    result = np.zeros_like(img_array)
    result[mask != 0] = [255, 255, 255]

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def detect_pose(image, line_thickness, circle_radius):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Convert PIL Image to OpenCV format
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process the image
    results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

    # Create a blank image to draw the pose on
    height, width, _ = img_array.shape
    pose_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            pose_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=circle_radius,
                circle_radius=circle_radius,
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=line_thickness
            ),
        )

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB))


def apply_blur(image, blur_amount):
    return image.filter(ImageFilter.GaussianBlur(radius=blur_amount))


def save_debug_image(img, original_path, new_name):
    debug_name = new_name.replace(Path(new_name).suffix, "-debug.png")
    debug_path = Path(original_path).parent / debug_name
    img.save(debug_path)
    print(f"Debug image saved: {debug_path}")


def truncate_file_name(file_name, hash_prefix, hash_str):
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    max_stem_length = (
        MAX_FILE_NAME_LENGTH
        - len(hash_prefix)
        - len(hash_str)
        - len(suffix)
        - 2
    )  # 2 for spaces

    if len(stem) > max_stem_length:
        truncated_stem = stem[: max_stem_length - 3] + "..."
    else:
        truncated_stem = stem

    new_name = f"{hash_prefix}{hash_str} {truncated_stem}{suffix}"

    # Ensure the new name is within the limit even after UTF-8 encoding
    while len(new_name.encode("utf-8")) > MAX_FILE_NAME_LENGTH:
        truncated_stem = truncated_stem[:-4] + "..."
        new_name = f"{hash_prefix}{hash_str} {truncated_stem}{suffix}"

    print(
        f"Debug: New file name length: {len(new_name.encode('utf-8'))} bytes"
    )
    return new_name


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Calculate phash and rename image files."
    )
    parser.add_argument("files", nargs="+", help="Image files to process")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually renaming files",
    )
    # Use with no options:
    # find . -type f -name '*.png' -exec imghash --blur 2.5 {} \;
    # Or with options:
    # imghash --mask {}
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Use rembg to generate an alpha mask and calculate phash on the inverted mask",
    )
    # imghash --outline {}
    parser.add_argument(
        "--outline",
        action="store_true",
        help="Use rembg to remove background and extract features from the subject",
    )
    # imghash --face {}
    # imghash --face --crop-padding 200 {}
    parser.add_argument(
        "--face",
        action="store_true",
        help="Detect and crop face before calculating phash",
    )
    # imghash --facial-features --crop-padding 200 --features-thickness 5 {}
    parser.add_argument(
        "--facial-features",
        action="store_true",
        help="Extract facial features before calculating phash",
    )
    # imghash --pose --pose-line-thickness 10 --pose-circle-radius 5 {}
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Detect pose and calculate phash on the pose skeleton",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediary image used for hash calculation",
    )
    parser.add_argument(
        "--blur",
        type=float,
        nargs="?",
        const=DEFAULT_BLUR,
        help=f"Apply Gaussian blur to the image before hashing. Optional blur amount (default: {DEFAULT_BLUR})",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=DEFAULT_CROP_PADDING,
        help=f"Number of pixels to add around the detected face (default: {DEFAULT_CROP_PADDING})",
    )
    parser.add_argument(
        "--pose-line-thickness",
        type=int,
        default=DEFAULT_POSE_LINE_THICKNESS,
        help=f"Thickness of the lines in the pose skeleton (default: {DEFAULT_POSE_LINE_THICKNESS})",
    )
    parser.add_argument(
        "--pose-circle-radius",
        type=int,
        default=DEFAULT_POSE_CIRCLE_RADIUS,
        help=f"Radius of the circles in the pose skeleton (default: {DEFAULT_POSE_CIRCLE_RADIUS})",
    )
    parser.add_argument(
        "--features-thickness",
        type=int,
        default=DEFAULT_FEATURES_THICKNESS,
        help=f"Thickness of the lines in feature extraction (default: {DEFAULT_FEATURES_THICKNESS})",
    )
    args = parser.parse_args()

    # Create a rembg session if --mask or --outline is used
    session = new_session("isnet-anime") if args.mask or args.outline else None

    # Iterate through the provided image file names
    for image_file in args.files:
        try:
            # Open the image file
            with Image.open(image_file) as img:
                if args.outline:
                    # Remove background
                    img = remove(img, session=session)
                    # Extract features
                    img = extract_features(img, args.features_thickness)
                    hash_prefix = f"phash-outline-t{args.features_thickness}-"
                elif args.pose:
                    # Detect pose
                    img = detect_pose(
                        img, args.pose_line_thickness, args.pose_circle_radius
                    )
                    hash_prefix = f"phash-pose-l{args.pose_line_thickness}c{args.pose_circle_radius}-"
                elif args.face or args.facial_features:
                    # Detect and crop face
                    img = detect_face(img, args.crop_padding)
                    if args.facial_features:
                        # Extract facial features
                        img = extract_features(img, args.features_thickness)
                    hash_prefix = f"phash-face-crop{args.crop_padding}-"
                    if args.facial_features:
                        hash_prefix = (
                            f"phash-face-features-t{args.features_thickness}-"
                            + hash_prefix
                        )
                elif args.mask:
                    # Generate alpha mask using rembg
                    mask = remove(
                        img,
                        only_mask=True,
                        post_process_mask=True,
                        session=session,
                    )
                    # Invert the mask
                    img = invert_mask(mask)
                    hash_prefix = "phash-mask-"
                else:
                    hash_prefix = "phash-"

                # Apply blur if specified
                if args.blur is not None:
                    img = apply_blur(img, args.blur)
                    hash_prefix = (
                        f"phash-blur{args.blur:.1f}-" + hash_prefix[6:]
                    )

                # Calculate the phash
                hash_value = imagehash.phash(img)

                # Convert the hash to a string
                hash_str = str(hash_value)

            # Use pathlib to handle file paths
            path = Path(image_file)
            original_name = path.name
            parent = path.parent

            # Create the new name, ensuring it's not too long
            new_name = truncate_file_name(original_name, hash_prefix, hash_str)

            # Create the new path
            new_path = parent / new_name

            # Save debug image if --debug flag is used
            if args.debug:
                save_debug_image(img, image_file, new_name)

            # Print the original and new basenames
            if args.dry_run:
                print(f"[DRY RUN] {original_name} -> {new_name}")
            else:
                # Rename the file
                shutil.move(str(path), str(new_path))
                print(f"{original_name} -> {new_name}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")


if __name__ == "__main__":
    main()
