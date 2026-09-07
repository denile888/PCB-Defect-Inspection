"""
Preprocessing - grayscale.

This is the melvinwongkakian preprocessing module, switched from colour to
single-channel grayscale to match the rest of the team (Lee Wanching's
homomorphic/Wiener/DWT chain and Lim Zseping's CLAHE/gamma/Laplacian chain
both operate on and return grayscale). Standardising on one channel depth
across all three of us means the cross-member comparison is measuring
technique quality, not colour-vs-grayscale as a confound.

This is a considered trade-off, not a free improvement: converting to
grayscale before saving means the copper/solder-mask colour distinction is
gone from the training images for this branch. Every function here still
returns a genuine single-channel (H, W) uint8 array, matching how the other
two members save their output, so a saved JPEG is truly grayscale rather
than a colour image with the channels merely made equal.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def to_gray(image_bgr):
    """Convert a BGR image to single-channel grayscale, or pass through."""
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def to_binary(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------
# A. Gaussian filtering
# ---------------------------------------------------------------------

def gaussian_filter(image, ksize=5, sigma=1.0):
    """Gaussian smoothing on the grayscale image.

    Converts to grayscale first, then blurs. The colour channels are
    discarded before any filtering happens, so the parameter values chosen
    in the Phase 2 study still apply - only the number of channels changed.
    """
    gray = to_gray(image)
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


# ---------------------------------------------------------------------
# B. Canny-guided edge enhancement
# ---------------------------------------------------------------------

def canny_edge_enhance(image, low=50, high=150, weight=0.5):
    """Canny edge detection, blended back into the grayscale base image.

    Both the base and the edge overlay are single-channel here, so the
    output stays grayscale throughout - unlike the colour version, which
    blended edges into the original BGR photograph.
    """
    gray = to_gray(image)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(smoothed, low, high, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    # Blend the edge map into the grayscale base, not the original colour
    # photograph - addWeighted on two single-channel arrays stays single
    # channel throughout.
    return cv2.addWeighted(gray, 1.0, edges, weight, 0)


# ---------------------------------------------------------------------
# C. Morphological enhancement
# ---------------------------------------------------------------------

def morphological_enhance(image, ksize=15):
    """Top-hat / black-hat contrast enhancement on the grayscale image.

        enhanced = gray + tophat - blackhat

    The colour version routed this through the LAB colour space so only
    lightness moved and hue stayed fixed. With a genuinely single-channel
    image there is no hue to preserve, so the LAB round-trip is dropped and
    the operation runs directly on gray.
    """
    gray = to_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # int16 avoids the wrap-around a uint8 subtraction would suffer if the
    # result went below 0.
    enhanced = gray.astype(np.int16) + tophat.astype(np.int16) - blackhat.astype(np.int16)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# D. Combined: Gaussian then Canny
# ---------------------------------------------------------------------

def gaussian_then_canny(image, ksize=9, sigma=2.5, low=50, high=150, weight=0.5):
    """The team's confirmed best-performing configuration.

    A3-style heavy Gaussian smoothing (ksize=9, sigma=2.5), followed by
    B2-style medium-strength Canny edge enhancement (low=50, high=150,
    weight=0.5). Chosen through a seed-verified comparative study (see
    03_run_comparison.ipynb) - it is not the strongest individual setting
    of either technique on its own, it is the specific COMBINATION that
    was measured to outperform A3 alone across multiple training seeds.

    Both stages operate on the same single-channel image, so this chains
    cleanly without a colour round-trip anywhere in the pipeline.
    """
    gray = to_gray(image)
    smoothed = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    edges = cv2.Canny(smoothed, low, high, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return cv2.addWeighted(smoothed, 1.0, edges, weight, 0)


# ---------------------------------------------------------------------
# Registry - mirrors the CONFIGS table used in 02_processed_dataset.ipynb
# ---------------------------------------------------------------------

TECHNIQUES = {
    "gaussian_light":  (gaussian_filter, dict(ksize=3, sigma=0.8)),
    "gaussian_medium": (gaussian_filter, dict(ksize=5, sigma=1.5)),
    "gaussian_heavy":  (gaussian_filter, dict(ksize=9, sigma=2.5)),
    "canny_weak":      (canny_edge_enhance, dict(low=100, high=200, weight=0.3)),
    "canny_medium":    (canny_edge_enhance, dict(low=50, high=150, weight=0.5)),
    "canny_strong":    (canny_edge_enhance, dict(low=30, high=90, weight=0.7)),
    "morph_small":     (morphological_enhance, dict(ksize=5)),
    "morph_medium":    (morphological_enhance, dict(ksize=15)),
    "morph_large":     (morphological_enhance, dict(ksize=25)),
    "winner (A3 heavy Gaussian + B2 medium Canny)": (gaussian_then_canny, dict()),
}

# The technique the app selects by default, applied automatically to every
# image unless someone picks something else from the sidebar for comparison.
# This is the one entry above that represents a conclusion, not just an
# option - it is the configuration the team's seed-verified comparative
# study found to outperform every individual technique on its own. Keeping
# it as a named constant, rather than a bare string repeated in app.py,
# means renaming or replacing the winner later only has one place to edit.
DEFAULT_TECHNIQUE = "winner (A3 heavy Gaussian + B2 medium Canny)"


def apply_technique(image_bgr, name, **overrides):
    """Look up a technique by name and apply it, honouring parameter overrides."""
    if name not in TECHNIQUES:
        raise ValueError(f"Unknown technique '{name}'. Choices: {list(TECHNIQUES)}")
    function, defaults = TECHNIQUES[name]
    params = {**defaults, **overrides}
    return function(image_bgr, **params)
