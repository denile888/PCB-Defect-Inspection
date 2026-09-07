"""
Object detection - bounding box and contour.

Bounding boxes come from the trained YOLO model (built separately, in the
comparison notebooks). This module adds the second half of the requirement:
a contour for the object inside each detected box.

Two modes, chosen automatically per call:

    reference available     -> difference-based contour (precise)
    no reference available  -> local-threshold contour (general-purpose)

The reference-based mode is the stronger of the two, and is available here
because every board in this dataset has a matching defect-free photograph
(PCB_USED). It finds the actual defect boundary, not just an edge inside the
box. The local-threshold fallback covers the case a real inspection line
would face: a new board with no stored reference image. Both return the same
result shape, so calling code does not need to know which mode ran.

Everything here operates in the SAME channel depth the box was detected in.
If detection ran on a grayscale-preprocessed image, pass grayscale in; the
functions do not silently reintroduce colour.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class ContourResult:
    """The contour found for one detected box, in full-image coordinates."""
    found: bool
    method: str                      # "reference_diff" | "local_threshold" | "none"
    contour: Optional[np.ndarray]    # Nx1x2 int32 points, or None
    area_px: float = 0.0
    perimeter_px: float = 0.0
    bbox_px: tuple = ()              # (x, y, w, h) of the contour itself


def _largest_contour(binary_mask, min_area):
    """Return the largest connected contour in a binary mask, or None."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    return contour


def _clean_mask(mask, close_size=5, open_size=3):
    """Close small gaps, then remove speckle - the same repair used in the
    edge-enhancement study, applied here to a segmentation mask instead of
    an edge map."""
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask


def find_contour(image_gray, box_xyxy, reference_gray=None, margin=12,
                 min_area=15, diff_threshold=None):
    """Find the object contour inside a detected bounding box.

    Parameters
    ----------
    image_gray
        The full board image, single channel, in the same pixel space the
        box coordinates were measured in.
    box_xyxy
        (x1, y1, x2, y2) in pixels - typically a YOLO detection box.
    reference_gray
        Optional defect-free version of the SAME board, pixel-aligned with
        image_gray. When given, the contour is found from the difference
        between the two, which isolates the actual defect rather than any
        edge that happens to fall inside the box.
    margin
        Pixels of context included around the box on every side. A crop
        exactly the size of the box would clip a contour that touches the
        box edge; the margin gives it room, and the search is still
        restricted to the largest contour so nearby unrelated features are
        not picked up.
    min_area
        Contours smaller than this many pixels are treated as noise and
        discarded, mirroring the area filter used throughout the edge
        enhancement study.
    diff_threshold
        Only used in reference mode. None means Otsu chooses it
        automatically. A crop with almost no real difference can still let
        Otsu split noise into two classes, so a floor is applied
        separately - see the guard below.

    Returns
    -------
    ContourResult, with the contour expressed in FULL-IMAGE coordinates.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    height, width = image_gray.shape[:2]

    cx1, cy1 = max(x1 - margin, 0), max(y1 - margin, 0)
    cx2, cy2 = min(x2 + margin, width), min(y2 + margin, height)

    crop = image_gray[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return ContourResult(found=False, method="none", contour=None)

    if reference_gray is not None:
        ref_crop = reference_gray[cy1:cy2, cx1:cx2]
        if ref_crop.shape == crop.shape:
            result = _contour_from_reference(crop, ref_crop, min_area, diff_threshold)
            if result.found:
                return _offset_result(result, cx1, cy1)
            # Reference gave nothing usable (e.g. genuinely no visible
            # difference in this crop) - fall through to local thresholding
            # rather than reporting no contour at all.

    result = _contour_from_local_threshold(crop, min_area)
    return _offset_result(result, cx1, cy1)


def _contour_from_reference(crop, ref_crop, min_area, diff_threshold):
    """Difference-based contour: isolates what changed versus the reference."""
    difference = cv2.absdiff(crop, ref_crop)

    # A crop with no real defect can still have Otsu split its noise into
    # two classes. Requiring a minimum peak difference before trusting Otsu
    # avoids manufacturing a contour out of nothing.
    if difference.max() < 15:
        return ContourResult(found=False, method="reference_diff", contour=None)

    if diff_threshold is None:
        threshold, mask = cv2.threshold(difference, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(difference, diff_threshold, 255, cv2.THRESH_BINARY)

    mask = _clean_mask(mask)
    contour = _largest_contour(mask, min_area)

    if contour is None:
        return ContourResult(found=False, method="reference_diff", contour=None)

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    bx, by, bw, bh = cv2.boundingRect(contour)
    return ContourResult(found=True, method="reference_diff", contour=contour,
                         area_px=area, perimeter_px=perimeter,
                         bbox_px=(bx, by, bw, bh))


def _contour_from_local_threshold(crop, min_area):
    """No-reference fallback: Otsu threshold on the crop itself.

    This is the general-purpose path - it works on a single image with
    nothing to compare against, at the cost of finding "whatever stands out
    locally" rather than "what actually differs from a known-good board".
    """
    threshold, mask = cv2.threshold(crop, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's foreground/background assignment is arbitrary; try both and
    # keep whichever produces the more plausible (smaller, more centred)
    # contour, since the defect is usually the minority region in the crop.
    inverse = cv2.bitwise_not(mask)

    candidates = []
    for candidate_mask in (mask, inverse):
        cleaned = _clean_mask(candidate_mask)
        contour = _largest_contour(cleaned, min_area)
        if contour is not None:
            candidates.append(contour)

    if not candidates:
        return ContourResult(found=False, method="local_threshold", contour=None)

    # Prefer the smaller of the two candidates: in a margin-padded crop, the
    # object of interest is usually the minority region, not the background.
    contour = min(candidates, key=cv2.contourArea)

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    bx, by, bw, bh = cv2.boundingRect(contour)
    return ContourResult(found=True, method="local_threshold", contour=contour,
                         area_px=area, perimeter_px=perimeter,
                         bbox_px=(bx, by, bw, bh))


def _offset_result(result, dx, dy):
    """Shift a crop-local contour result back into full-image coordinates."""
    if not result.found:
        return result
    shifted = result.contour + np.array([dx, dy])
    bx, by, bw, bh = result.bbox_px
    return ContourResult(found=True, method=result.method, contour=shifted,
                         area_px=result.area_px, perimeter_px=result.perimeter_px,
                         bbox_px=(bx + dx, by + dy, bw, bh))


def draw_contour(image_bgr, result, color=(0, 255, 255), thickness=2):
    """Draw a ContourResult onto a colour image for display, in place."""
    if result.found and result.contour is not None:
        cv2.drawContours(image_bgr, [result.contour], -1, color, thickness)
    return image_bgr


def find_all_defects_by_reference(image_gray, reference_gray, min_area=15,
                                  merge_distance=15, sensitivity=99.9,
                                  min_threshold=12):
    """Classical whole-board defect flagging, with no trained model at all.

    This does the same differencing as find_contour's reference mode, but
    across the entire board rather than inside a pre-supplied box, so it can
    stand in as the detection stage when no YOLO weights are loaded. It is
    a fallback for demonstrating the app end to end, not a replacement for
    the trained detector - it has no notion of defect class and will not
    match a trained model's accuracy.

    Otsu's threshold is unreliable here, and fails in a specific,
    measured way: on a real mouse_bite test image it chose a threshold of
    1 (out of 255), which classified 34% of the entire board as "different"
    and returned a single contour the size of the whole image. `sensitivity`
    (a percentile of the difference distribution) and `min_threshold` are
    both tried as alternatives, and the OPERATING THRESHOLD IS WHICHEVER OF
    THE THREE IS HIGHEST - not a union of masks. A union can only make a
    mask more permissive, which is the wrong direction when the failure
    mode is Otsu already being too permissive; only taking the strictest
    candidate actually guards against it.
    """
    if image_gray.shape != reference_gray.shape:
        reference_gray = cv2.resize(reference_gray,
                                    (image_gray.shape[1], image_gray.shape[0]))

    difference = cv2.absdiff(image_gray, reference_gray)

    otsu_value, _ = cv2.threshold(difference, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    percentile_value = float(np.percentile(difference, sensitivity))

    operating_threshold = max(otsu_value, percentile_value, min_threshold)
    _, mask = cv2.threshold(difference, operating_threshold, 255, cv2.THRESH_BINARY)
    mask = _clean_mask(mask, close_size=9, open_size=3)

    # Dilating before finding components merges fragments of the same
    # defect that the threshold split apart, without merging genuinely
    # separate defects that are far enough apart.
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (merge_distance, merge_distance))
    merged = cv2.dilate(mask, merge_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)

    results = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        # Re-run the tight, un-dilated mask inside just this component's
        # box, so the reported contour is the real defect outline rather
        # than the merge-dilated blob.
        pad = merge_distance
        box = (x - pad, y - pad, x + w + pad, y + h + pad)
        result = find_contour(image_gray, box, reference_gray=reference_gray,
                              margin=0, min_area=min_area)
        if result.found:
            results.append(result)

    return results
