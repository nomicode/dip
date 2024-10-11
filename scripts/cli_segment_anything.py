#!/usr/bin/env python3

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer, Group
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_SAM_CHECKPOINT = (
    "/Volumes/External HD/Models/Segment Anything/sam_vit_b_01ec64.pth"
)


def process_image(image, sam_checkpoint):
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Ensure image is in RGBA mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Process the original image
    img_array = np.array(image)
    masks = mask_generator.generate(img_array[:, :, :3])  # SAM expects RGB

    # Sort masks by area (largest to smallest)
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    return sorted_masks, image


def cluster_segments(masks, n_clusters=5):
    # Extract features for clustering (e.g., center coordinates and area)
    features = []
    for mask in masks:
        y, x = np.where(mask["segmentation"])
        center_y, center_x = y.mean(), x.mean()
        area = mask["area"]
        features.append([center_x, center_y, area])

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_features)

    return cluster_labels


def create_psd_document(image, masks, output_path, n_clusters=5):
    # Create a new PSD file
    psd = PSDImage.new(
        mode="RGBA",
        size=(image.width, image.height),
    )

    # Add the original image as the background layer
    bg_layer = PixelLayer.frompil(image, psd, name="Background")
    bg_layer.visible = False
    psd.append(bg_layer)

    # Convert image to numpy array
    img_array = np.array(image)

    # Cluster segments
    cluster_labels = cluster_segments(masks, n_clusters)

    # Create all layers first
    all_layers = []
    for i, mask in enumerate(masks):
        mask_array = mask["segmentation"]

        # Create a new RGBA array for the segment
        segment_array = np.zeros(
            (image.height, image.width, 4), dtype=np.uint8
        )

        # Copy the original image data
        segment_array[:, :, :3] = img_array[:, :, :3]

        # Set the alpha channel using the mask
        segment_array[:, :, 3] = mask_array * 255

        # Convert to PIL Image
        segment_image = Image.fromarray(segment_array, mode="RGBA")

        # Create a new layer with the masked image
        segment_layer = PixelLayer.frompil(
            segment_image, psd, name=f"Segment {i+1}"
        )
        all_layers.append(segment_layer)

    # Create an empty main group
    main_group = Group.new(name="Segment Anything", parent=psd)

    # Create cluster groups and add layers to them
    for i in range(n_clusters):
        cluster_layers = [
            layer
            for layer, label in zip(all_layers, cluster_labels)
            if label == i
        ]
        Group.group_layers(
            cluster_layers, name=f"Cluster {i+1}", parent=main_group
        )

    # Save the PSD file
    psd_path = output_path.with_suffix(".psd")
    psd.save(psd_path)
    print(f"PSD file saved: {psd_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment an image using Segment Anything Model (SAM) and save as a PSD file with layers."
    )
    parser.add_argument("image", help="Image file to process")
    parser.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_SAM_CHECKPOINT,
        help=f"Path to the SAM checkpoint file (default: {DEFAULT_SAM_CHECKPOINT})",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters for grouping segments (default: 5)",
    )

    args = parser.parse_args()

    input_path = Path(args.image)
    with Image.open(input_path) as img:
        masks, original_img = process_image(img, args.sam_checkpoint)
        output_path = input_path.with_stem(
            f"{input_path.stem}_segment_anything"
        )
        create_psd_document(original_img, masks, output_path, args.clusters)


if __name__ == "__main__":
    main()

# Commented out plotting code:
"""
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Debug output
for i, mask in enumerate(masks):
    print(f"mask[{i}]: predicted_iou = {mask['predicted_iou']}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.show()
"""
