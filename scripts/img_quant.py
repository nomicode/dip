#!/usr/bin/env python3

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2  # Ensure cv2 is imported for color conversion


def adaptive_posterization(image):
    """
    Perform adaptive posterization on the input image.
    """

    def adaptive_posterize(image, num_levels=8, edge_threshold=100):
        # Convert image to grayscale for edge detection
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Create a mask where edges are detected
        edge_mask = edges > edge_threshold

        # Convert image to LAB color space
        lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)

        # Initialize the result image
        result = np.zeros_like(lab_image)

        for i in range(3):  # For each channel in LAB
            channel = lab_image[:, :, i]

            # Compute histogram and posterization mapping
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            cdf = hist.cumsum()
            cdf_normalized = (
                cdf * (255.0 / cdf[-1]) if cdf[-1] != 0 else cdf
            )  # Avoid division by zero
            posterize_map = np.floor(cdf_normalized / (256 / num_levels)) * (
                256 / num_levels
            )

            # Apply posterization
            posterized = posterize_map[
                channel.astype(int)
            ]  # Ensure channel is int for indexing

            # Preserve original values at edge locations
            posterized[edge_mask] = channel[edge_mask]

            result[:, :, i] = posterized

        # Convert back to RGB color space
        result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return Image.fromarray(result_rgb)

    return adaptive_posterize(image)


def adaptive_posterization(image):
    """
    Perform adaptive posterization on the input image.

    This method adjusts the quantization levels based on local image content,
    including edge information.

    Implementation:
    1. Detect edges using Canny edge detection
    2. Create a mask where edges are detected
    3. Apply different levels of posterization based on edge presence
    """

    def adaptive_posterize(image, num_levels=16, edge_threshold=100):
        # Convert image to grayscale for edge detection
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Create a mask where edges are detected
        edge_mask = edges > edge_threshold
        # Image.fromarray(edge_mask).show()

        # Convert image to LAB color space
        lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)

        # Initialize the result image
        result = np.zeros_like(lab_image)

        for i in range(3):  # For each channel in LAB
            channel = lab_image[:, :, i]

            # Simple quantization
            posterized = np.floor(channel / (256 / num_levels)) * (256 / num_levels)

            # Preserve original values at edge locations
            posterized[edge_mask] = channel[edge_mask]

            result[:, :, i] = posterized

        # Convert back to RGB color space
        result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return Image.fromarray(result_rgb)

    return adaptive_posterize(image)


from skimage.segmentation import slic
import numpy as np
from PIL import Image


def content_aware_posterization(image):
    """
    Perform content-aware posterization on the input image.
    """

    # TODO: Run this multiple times (e.g., using n_segments=10000, or 100, and
    # try a range of values like this)
    def content_aware_posterize(image, n_segments=(100), compactness=10, num_levels=16):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Perform SLIC segmentation
        segments = slic(img_array, n_segments=n_segments, compactness=compactness)

        # Initialize result array
        result = np.zeros_like(img_array)

        for segment_id in np.unique(segments):
            # Create mask for current segment
            mask = segments == segment_id

            # Get colors in the current segment
            segment_colors = img_array[mask]

            # Compute color histogram for the segment
            hist, edges = np.histogramdd(
                segment_colors,
                bins=num_levels,
                range=[(0, 255), (0, 255), (0, 255)],
            )

            # Find the most common color in the segment
            most_common_index = np.unravel_index(hist.argmax(), hist.shape)

            # Convert the index back to RGB color
            most_common_color = [
                int(edges[0][most_common_index[0]]),  # R
                int(edges[1][most_common_index[1]]),  # G
                int(edges[2][most_common_index[2]]),  # B
            ]

            # Assign the most common color to all pixels in the segment
            result[mask] = most_common_color

        return Image.fromarray(result.astype(np.uint8))

    return content_aware_posterize(image)


def bilateral_filtering_posterization(image):
    """
    Perform bilateral filtering with posterization on the input image.

    This method combines bilateral filtering (which preserves edges while smoothing)
    with posterization.

    Implementation:
    1. Apply bilateral filtering to smooth non-edge areas
    2. Detect edges in the original image
    3. Posterize the filtered image
    4. Blend the posterized result with the original edges
    """

    def bilateral_filter_posterize(
        image, d=9, sigma_color=75, sigma_space=75, num_levels=16
    ):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Apply bilateral filtering
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # Detect edges in the original image
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Posterize the filtered image
        posterized = np.floor_divide(filtered, 256 // num_levels) * (256 // num_levels)

        # Blend posterized result with original edges
        result = np.where(edges[:, :, None] > 0, img_array, posterized)

        return Image.fromarray(result.astype(np.uint8))

    return bilateral_filter_posterize(image)


def gradient_preserving_quantization(image):
    """
    Perform gradient-preserving quantization on the input image.

    This method aims to maintain the gradient information in the image while reducing colors.

    Implementation:
    1. Compute image gradients
    2. Quantize colors while minimizing changes to the gradient field
    3. Use k-means clustering to find the best color palette that preserves gradients
    """

    def gradient_preserving_quantize(image, num_colors=16):
        img_array = np.array(image)

        # Compute gradients
        gx = np.gradient(img_array, axis=0)
        gy = np.gradient(img_array, axis=1)

        # Flatten the image and gradients
        pixels = img_array.reshape(-1, 3)
        gradients = np.concatenate((gx.reshape(-1, 3), gy.reshape(-1, 3)), axis=1)

        # Use k-means to find color centers
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels, sample_weight=np.linalg.norm(gradients, axis=1))

        # Map each pixel to the nearest center
        quantized = kmeans.cluster_centers_[kmeans.labels_]

        return Image.fromarray(quantized.reshape(img_array.shape).astype(np.uint8))

    return gradient_preserving_quantize(image)


def superpixel_based_posterization(image):
    """
    Perform superpixel-based posterization on the input image.

    This method segments the image into superpixels before posterization.

    Implementation:
    1. Generate superpixels (which tend to adhere to edges)
    2. Compute the average color for each superpixel
    3. Posterize each superpixel independently
    """

    def superpixel_posterize(image, n_segments=100, compactness=10, num_levels=16):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Generate superpixels
        segments = slic(img_array, n_segments=n_segments, compactness=compactness)

        # Compute average color for each superpixel
        averaged = label2rgb(segments, img_array, kind="avg")

        # Posterize the averaged image
        posterized = np.floor_divide(averaged, 256 // num_levels) * (256 // num_levels)

        return Image.fromarray(posterized.astype(np.uint8))

    return superpixel_posterize(image)


def edge_detection_quantization(image):
    """
    Perform edge detection in quantization on the input image.

    This method preserves edges while quantizing colors in the image.

    Implementation:
    1. Detect edges in the image
    2. Apply color quantization to non-edge regions
    3. Preserve original colors or apply less aggressive quantization at edge regions
    """

    def edge_preserving_quantize(image, num_colors=16, edge_threshold=100):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Detect edges
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Create edge mask
        edge_mask = edges > edge_threshold

        # Quantize non-edge regions
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels[~edge_mask.flatten()])

        # Apply quantization
        quantized = kmeans.predict(pixels)
        result = kmeans.cluster_centers_[quantized].reshape(img_array.shape)

        # Preserve original colors at edge regions
        result[edge_mask] = img_array[edge_mask]

        return Image.fromarray(result.astype(np.uint8))

    return edge_preserving_quantize(image)


def k_means_color_quantization(image):
    """
    Perform k-means color quantization on the input image.

    This method uses the k-means clustering algorithm to reduce the number of colors in the image.

    Implementation:
    1. Reshape the image into a list of pixels
    2. Apply k-means clustering to find color centers
    3. Replace each pixel with its nearest color center
    """

    def kmeans_quantize(image, n_colors=16):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Reshape the image
        pixels = img_array.reshape(-1, 3)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)

        # Replace each pixel with its nearest center
        quantized = kmeans.cluster_centers_[kmeans.labels_]

        # Reshape back to image dimensions
        quantized_img = quantized.reshape(img_array.shape)

        return Image.fromarray(quantized_img.astype(np.uint8))

    return kmeans_quantize(image)


def homebrew_k_means_quantization(image):
    """
    Perform a modified k-means quantization technique on the input image.

    This method uses k-means clustering to identify color clusters and then
    allocates a fixed number of colors to each cluster, allowing for better
    color preservation across different areas of the image.

    Implementation steps:
    1. Perform k-means clustering on the image colors
    2. For each cluster, perform secondary clustering to find representative colors
    3. Assign each pixel to the nearest color within its cluster
    """

    # FIXME: These do not work as expected
    n_color_clusters = 8
    n_cluster_colors = 8

    def homebrew_kmeans_quantize(image, n_clusters=n_color_clusters):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Reshape the image
        pixels = img_array.reshape(-1, 3)

        # Perform initial k-means clustering
        kmeans = KMeans(n_clusters=n_cluster_colors, random_state=42)
        labels = kmeans.fit_predict(pixels)

        # Initialize the result array
        result = np.zeros_like(pixels)

        for i in range(n_clusters):
            # Get pixels belonging to this cluster
            cluster_pixels = pixels[labels == i]

            if len(cluster_pixels) == 0:
                continue

            # Perform a secondary k-means within this cluster
            sub_kmeans = KMeans(n_clusters=n_cluster_colors, random_state=42)
            sub_labels = sub_kmeans.fit_predict(cluster_pixels)

            # Assign quantized colors to pixels in this cluster
            result[labels == i] = sub_kmeans.cluster_centers_[sub_labels]

        # Reshape back to image dimensions
        quantized_img = result.reshape(img_array.shape)

        return Image.fromarray(quantized_img.astype(np.uint8))

    return homebrew_kmeans_quantize(image)


def display_image(image_array, title="Image"):
    """Utility function to display an image."""
    plt.imshow(image_array)
    plt.axis("off")
    plt.title(title)
    plt.show()


def plot_centroids(primary_centroids, secondary_centroids_list):
    """Plot primary and secondary centroids as color swatches."""
    num_primary = primary_centroids.shape[0]
    max_secondary = max(len(sec) for sec in secondary_centroids_list)

    # Create a figure to hold the swatches with a smaller size
    fig, ax = plt.subplots(
        num_primary, max(1, max_secondary + 1), figsize=(4, 1.5 * num_primary)
    )

    # Plot primary centroids
    for i in range(num_primary):
        # Convert LAB to RGB for display
        primary_rgb = cv2.cvtColor(
            np.uint8([[primary_centroids[i]]]), cv2.COLOR_LAB2RGB
        )[0][0]
        ax[i, 0].imshow(
            np.full((20, 20, 3), primary_rgb / 255.0)
        )  # Smaller swatch size
        ax[i, 0].axis("off")
        ax[i, 0].set_title(f"Primary Centroid {i}", fontsize=8)

        # Plot secondary centroids to the right of the primary centroid
        for j in range(len(secondary_centroids_list[i])):
            # Convert LAB to RGB for display
            secondary_rgb = cv2.cvtColor(
                np.uint8([[secondary_centroids_list[i][j]]]), cv2.COLOR_LAB2RGB
            )[0][0]
            ax[i, j + 1].imshow(
                np.full((20, 20, 3), secondary_rgb / 255.0)
            )  # Smaller swatch size
            ax[i, j + 1].axis("off")

    # Hide any unused subplots
    for i in range(num_primary):
        for j in range(len(secondary_centroids_list[i]), max_secondary + 1):
            ax[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def homebrew_k_means_quantization_v2(image):

    def homebrew_kmeans_quantize_v2(image, primary_k=10, secondary_k=8):
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        # Normalize pixel values to [0, 1]
        pixels = lab_image.reshape(-1, 3).astype(np.float32)

        # Primary K-Means clustering in LAB space with improved initialization
        kmeans_primary = KMeans(n_clusters=primary_k, n_init=30, init="k-means++")
        primary_labels = kmeans_primary.fit_predict(pixels)

        # Display primary clustering results
        primary_centroids = kmeans_primary.cluster_centers_  # No scaling needed in LAB
        print("Primary Centroids (LAB):", primary_centroids)

        # Initialize an empty image for the quantized output
        result = np.zeros_like(pixels)

        # Store secondary centroids for plotting
        secondary_centroids_list = []

        # Loop through each primary cluster
        for i in range(primary_k):
            # Create a mask for the current primary cluster
            mask_primary = primary_labels == i

            # Get pixels belonging to the current primary cluster
            cluster_pixels = pixels[mask_primary]  # Use the mask directly

            if len(cluster_pixels) > 0:
                kmeans_secondary = KMeans(
                    n_clusters=secondary_k, n_init=30, init="k-means++"
                )
                secondary_labels = kmeans_secondary.fit_predict(cluster_pixels)
                secondary_centroids = (
                    kmeans_secondary.cluster_centers_
                )  # No scaling needed in LAB
                print(
                    f"Secondary Centroids for Cluster {i} (LAB):", secondary_centroids
                )

                # Store secondary centroids for plotting
                secondary_centroids_list.append(secondary_centroids)

                for j in range(secondary_k):
                    mask_secondary = secondary_labels == j

                    # Create a full mask for the secondary cluster that matches the original pixels
                    full_mask_secondary = np.zeros_like(primary_labels, dtype=bool)
                    full_mask_secondary[mask_primary] = (
                        mask_secondary  # Assign the secondary mask only where the primary mask is True
                    )

                    # Combine the masks
                    combined_mask = mask_primary & full_mask_secondary
                    result[combined_mask] = secondary_centroids[j]

        # Reshape back to image dimensions
        quantized_img_lab = result.reshape(img_array.shape)

        # Convert the quantized LAB image back to RGB
        quantized_img_rgb = cv2.cvtColor(
            quantized_img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB
        )

        # Ensure the pixel values are in the correct range
        quantized_img_rgb = np.clip(quantized_img_rgb, 0, 255).astype(np.uint8)

        # Display the final quantized image
        # display_image(quantized_img_rgb, title="Quantized Image")

        # Debug plot for centroids
        # plot_centroids(primary_centroids, secondary_centroids_list)

        return Image.fromarray(quantized_img_rgb)

    return homebrew_kmeans_quantize_v2(image)


# TODO

# <user> is there any way to make sure that the k-means sampling catches the
# outlyers? in my head, conceptually, I imagine boosting the contrast all the
# way up, which would pull out all the contrasting details, then running the
# k-means on that modified image, but taking the results (centroids), and
# projecting them back on to the original image to continue doing the anlysis
# but maybe there is a better way to address this issue, or maybe there is an
# entirely different technique (something that isn't k-means) which will
# attempt to sample the image to produce a palette where the colors chosen are
# maximally far apart (i.e., maximizes the distance between colors), which
# should pull out the contrasting elements thoughts?

# <assistant> Your idea of enhancing the contrast to highlight outliers before
# applying k-means is a valid approach and can help in identifying important
# details in the image. However, there are also alternative techniques and
# strategies that can be employed to achieve better color quantization and
# ensure that contrasting colors are captured effectively.
#
# Strategies to Capture Outliers and Enhance Color Quantization
#
# - Contrast Enhancement:
#   - Histogram Equalization: This technique can enhance the contrast of the
#     image, making the outliers more pronounced. You can apply histogram
#     equalization in the LAB color space to enhance the L (lightness) channel.
#   - Adaptive Histogram Equalization (CLAHE): This method improves local
#     contrast and can be particularly effective in images with varying
#     lighting conditions.
#
# - Using Different Clustering Techniques:
#   - DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
#     This algorithm can identify clusters of varying shapes and sizes and is
#     particularly good at finding outliers. It does not require specifying the
#     number of clusters in advance.
#   - Mean Shift Clustering: This method identifies clusters based on the
#     density of data points and can adaptively find the number of clusters
#     based on the data distribution.
#
# - Maximizing Color Distance:
#   - Color Space Sampling: Instead of using k-means, you can sample colors
#     from the image in a way that maximizes the distance between them. This
#     can be done using techniques like:
#   - Farthest Point Sampling: Start with a random color and iteratively select
#     the color that is farthest from the already selected colors.
#   - Diversity Sampling: Use a method that selects colors based on their
#     distance in color space, ensuring that selected colors are maximally
#     distinct.
#
# - Combining Techniques:
#   - You can combine contrast enhancement with k-means or other clustering
#     techniques. For example, enhance the contrast of the image, run k-means
#     to find the centroids, and then project those centroids back onto the
#     original image.

# Example Implementation
# Here’s a conceptual implementation that combines contrast enhancement with k-means clustering:


def enhance_contrast(image):
    """Enhance contrast using CLAHE."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge back the channels
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_image


def homebrew_k_means_quantization_v3(image):

    def homebrew_kmeans_quantize_v3(image, n_colors=16):
        """Perform k-means color quantization on the image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Enhance contrast
        enhanced_image = enhance_contrast(img_array)

        # Reshape the image for k-means
        pixels = enhanced_image.reshape(-1, 3)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_colors, n_init=30, init="k-means++")
        kmeans.fit(pixels)

        # Get the centroids
        centroids = kmeans.cluster_centers_.astype(int)

        # Create a quantized image
        quantized_image = centroids[kmeans.labels_].reshape(
            img_array.shape
        )  # Use img_array.shape

        return Image.fromarray(quantized_image)

    return homebrew_kmeans_quantize_v3(image)


# TODO

# <user> is there an established technique or library that focuses on
# quantizing images based on perceptual qualities? (i.e. taylored to human
# perception)

# <assistant> Yes, there are several established techniques and libraries that
# focus on perceptual image quantization, which takes into account human visual
# perception to produce more visually appealing results. Here are some notable
# methods and libraries:
#
# 1. Perceptual Color Quantization Techniques
#    - CIEDE2000 Color Difference: This is a color difference formula that is
#      more aligned with human perception than traditional RGB distance
#      metrics. It can be used in clustering algorithms to ensure that colors
#      that are perceptually similar are treated as such.
#    - K-Means in LAB Color Space: As previously mentioned, using k-means
#      clustering in the LAB color space can help because LAB is designed to be
#      perceptually uniform. This means that the distance between colors in LAB
#      space corresponds more closely to how humans perceive color differences.
#
# 2. Libraries and Tools
#    - ImageMagick: This powerful image manipulation library includes tools for
#      color quantization that are perceptually aware. It uses algorithms like
#      the Octree quantization method, which can produce good results for
#      reducing the number of colors in an image while maintaining visual
#      quality.
#    - Pillow (PIL): The Python Imaging Library (Pillow) has built-in methods
#      for quantizing images. The Image.quantize() method uses a median cut
#      algorithm, which can be effective for perceptual color quantization.
#    - OpenCV: OpenCV provides various functions for color space conversion and
#      clustering, which can be combined with perceptual metrics for
#      quantization.
#    - ColorQuant: This is a Python library specifically designed for color
#      quantization. It uses perceptual metrics to ensure that the resulting
#      palette is visually appealing. It implements algorithms like the median
#      cut and octree quantization.
#    - Quantize: This is a library that implements various color quantization
#      algorithms, including those that are perceptually aware. It can be used
#      to reduce the number of colors in an image while maintaining visual
#      fidelity.
#
# 3. Deep Learning Approaches
#    - Neural Networks for Color Quantization: Some research has focused on
#      using neural networks to learn perceptual color quantization. These
#      methods can be more complex but can yield excellent results by training
#      on large datasets to understand human color perception.


# Perceptual quantization demo
def perceptual_quantization(image):

    def _perceptual_quantization(image, num_colors=16):
        # Convert the image to P mode (palette mode)
        quantized_image = image.quantize(colors=num_colors)

        return quantized_image

    return _perceptual_quantization(image)


def process_image(input_path):
    """Process the input image using various quantization methods."""
    input_image = Image.open(input_path)

    quantization_methods = {
        "adaptive": adaptive_posterization,
        "content-aware": content_aware_posterization,
        "bilateral": bilateral_filtering_posterization,
        "gradient": gradient_preserving_quantization,
        "superpixel": superpixel_based_posterization,
        "edge": edge_detection_quantization,
        "k-means": k_means_color_quantization,
        "homebrew-k-means": homebrew_k_means_quantization,
        "homebrew-k-means-v2": homebrew_k_means_quantization_v2,
        # FIXME: This is broken
        # "homebrew-k-means-v3": homebrew_k_means_quantization_v3,
        "perceptual_quantization": perceptual_quantization,
    }

    for method_name, method_func in quantization_methods.items():
        output_image = method_func(input_image)
        output_path = input_path.with_name(
            f"{input_path.stem} img_quant-{method_name}{input_path.suffix}"
        )
        output_image.save(output_path)
        print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Image Quantization Tool")
    parser.add_argument("input_image", type=str, help="Path to the input image file")
    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist.")
        return

    process_image(input_path)


if __name__ == "__main__":
    main()
