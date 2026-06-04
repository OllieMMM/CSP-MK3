import cv2 as cv
import numpy as np


WOODEN = np.array([
    [70, 76, 156],
    [119, 170, 230],
    [35, 2, 105],
    [167, 205, 235],
    [128, 160, 189],
    [80, 127, 163]
], dtype=np.uint8)


def image_to_palette_indices(img: np.ndarray,
                             palette: np.ndarray) -> np.ndarray:
    """
    Converts a palette-quantized BGR image into an index image.

    Returns:
        uint8 array of shape (H, W)
    """

    h, w, _ = img.shape

    index_map = np.zeros((h, w), dtype=np.uint8)

    for idx, color in enumerate(palette):

        mask = np.all(img == color, axis=2)

        index_map[mask] = idx

    return index_map


def palette_indices_to_image(index_map: np.ndarray,
                             palette: np.ndarray) -> np.ndarray:
    """
    Converts an index image back into BGR.
    """

    return palette[index_map]


def dominant_neighbor(index_map: np.ndarray,
                      region_mask: np.ndarray,
                      current_idx: int):
    """
    Finds the dominant neighbouring palette index.
    """

    kernel = np.ones((3, 3), np.uint8)

    dilated = cv.dilate(
        region_mask.astype(np.uint8),
        kernel,
        iterations=1
    ).astype(bool)

    border = dilated & (~region_mask)

    neighbors = index_map[border]

    neighbors = neighbors[
        neighbors != current_idx
    ]

    if len(neighbors) == 0:
        return None

    labels, counts = np.unique(
        neighbors,
        return_counts=True
    )

    return labels[np.argmax(counts)]


def merge_small_regions(index_map: np.ndarray,
                        min_area: int = 50,
                        connectivity: int = 8):
    """
    Remove regions smaller than min_area.
    """

    output = index_map.copy()

    num_palette_colors = int(output.max()) + 1

    for palette_idx in range(num_palette_colors):

        mask = (output == palette_idx)

        num_labels, labels, stats, _ = \
            cv.connectedComponentsWithStats(
                mask.astype(np.uint8),
                connectivity=connectivity
            )

        for cc_id in range(1, num_labels):

            area = stats[
                cc_id,
                cv.CC_STAT_AREA
            ]

            if area >= min_area:
                continue

            region_mask = labels == cc_id

            replacement = dominant_neighbor(
                output,
                region_mask,
                palette_idx
            )

            if replacement is None:
                continue

            output[region_mask] = replacement

    return output


def cleanup_quantized_image(
        img: np.ndarray,
        palette: np.ndarray,
        min_area: int = 50,
        passes: int = 3):
    """
    Full cleanup pipeline.
    """

    index_map = image_to_palette_indices(
        img,
        palette
    )

    for _ in range(passes):

        index_map = merge_small_regions(
            index_map,
            min_area=min_area
        )

    cleaned = palette_indices_to_image(
        index_map,
        palette
    )

    return cleaned


if __name__ == "__main__":

    img = cv.imread("Image Processing/filteredImage.png")

    print(img.shape)

    cleaned = cleanup_quantized_image(
        img,
        WOODEN,
        min_area=50,
        passes=3
    )

    cv.imwrite(
        "filteredImage_cleaned.png",
        cleaned
    )

    # This implamentation is slow, but does a good job at preserving edges and removing small artifacts from the image!