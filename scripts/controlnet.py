#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from controlnet_aux.processor import Processor

CONTROLNET_MODELS = [
    "canny",
    "depth_leres",
    "depth_leres++",
    "depth_midas",
    "depth_zoe",
    "lineart_anime",
    "lineart_coarse",
    "lineart_realistic",
    "mediapipe_face",
    "mlsd",
    "normal_bae",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "scribble_hed",
    "scribble_pidinet",
    "shuffle",
    "softedge_hed",
    "softedge_hedsafe",
    "softedge_pidinet",
    "softedge_pidsafe",
    # "dwpose",
]

DEFAULT_CONTROLNET_MODEL = "canny"
DEFAULT_LINE_THICKNESS = 1


def apply_controlnet(image, model_name, line_thickness):
    processor = Processor(model_name)
    result = processor(image, to_pil=True)

    if line_thickness > 1 and model_name in [
        "lineart_anime",
        "lineart_coarse",
        "lineart_realistic",
    ]:
        result = thicken_lines(result, line_thickness)

    return result


def thicken_lines(image, thickness):
    # Convert to numpy array
    img_array = np.array(image)

    # Find all non-white pixels
    y, x = np.where(img_array[:, :, 0] < 255)

    # Create a new white image
    new_img = Image.new("RGB", image.size, (255, 255, 255))
    draw = ImageDraw.Draw(new_img)

    # Draw thicker lines
    for i in range(len(x)):
        draw.ellipse(
            [
                x[i] - thickness // 2,
                y[i] - thickness // 2,
                x[i] + thickness // 2,
                y[i] + thickness // 2,
            ],
            fill=(0, 0, 0),
        )

    return new_img


class ControlnetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in CONTROLNET_MODELS:
            raise argparse.ArgumentError(
                self,
                f"invalid choice: '{values}' (choose from {', '.join(CONTROLNET_MODELS)})",
            )
        setattr(namespace, self.dest, values)


def save_image(image, path):
    image.save(path)
    print(f"Image saved: {path}")


def get_model_name(args_model):
    if args_model:
        return args_model
    env_model = os.environ.get("CONTROLNET_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    return DEFAULT_CONTROLNET_MODEL


def process_image(input_path, model_name, line_thickness):
    with Image.open(input_path) as img:
        controlnet_image = apply_controlnet(img, model_name, line_thickness)
        output_path = input_path.with_stem(
            f"{input_path.stem}_controlnet-{model_name}_thickness-{line_thickness}"
        ).with_suffix(".png")
        save_image(controlnet_image, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Apply ControlNet processing to an image."
    )
    parser.add_argument("image", help="Image file to process")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        action=ControlnetAction,
        help=f"ControlNet model to apply (default: {DEFAULT_CONTROLNET_MODEL}, can also be set via CONTROLNET_MODEL environment variable)",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Process the image with all valid ControlNet models",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=DEFAULT_LINE_THICKNESS,
        help=f"Thickness of lines for applicable models (default: {DEFAULT_LINE_THICKNESS})",
    )

    args = parser.parse_args()

    input_path = Path(args.image)

    if args.demo:
        for model in CONTROLNET_MODELS:
            process_image(input_path, model, args.line_thickness)
    else:
        model_name = get_model_name(args.model)
        process_image(input_path, model_name, args.line_thickness)


if __name__ == "__main__":
    main()
