"""
Teammates' preprocessing techniques.

Lee Wan Ching and Lim Sze Ping each implemented a different classical
image processing technique for this project, per Assignment Mode A (each
member compares a different technique against the others). The functions
and parameter sets below are reproduced from their own notebooks:

    Lee Wan Ching - notebook/leewanching/pipeline_leewanching.ipynb
    Lim Sze Ping  - notebook/limszeping/prepare_limszeping.ipynb
                    (cell 10 - the one that actually built the real
                    dataset; an earlier exploratory cell in the same
                    notebook defines a different, abandoned parameter set)

melvinwongkakian's own techniques live in core/preprocessing.py; keeping
this in a separate file means the three are never confused for one
person's work, and each algorithm can be traced back to its source
unambiguously.

Both pipelines are reproduced as written, with one addition: a shape
safety check after each pipeline runs (see _match_original_shape below).
Lee Wan Ching's wavelet step in particular can shift the image size by a
single pixel on odd dimensions - confirmed on a real board in this dataset
(1921x2904 becomes 1922x2904) - which would silently misalign every
bounding box drawn on top of the result. This correction is not part of
either person's original algorithm; it exists only because this code now
runs inside a live app where box coordinates must stay valid.
"""

import cv2
import numpy as np
import pywt
from scipy.signal import wiener


def _match_original_shape(image, original_shape):
    """Crop or pad back to the exact original height/width if a filtering
    step shifted it - see the module docstring for why this is needed and
    is not part of either teammate's original algorithm.
    """
    target_h, target_w = original_shape[:2]
    h, w = image.shape[:2]
    if (h, w) == (target_h, target_w):
        return image

    cropped = image[:target_h, :target_w]
    if cropped.shape[:2] == (target_h, target_w):
        return cropped

    # Only reached if the mismatch went the other way (image smaller than
    # the target) - not observed in testing, but padding rather than
    # crashing keeps this safe if it ever does happen.
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)


# =======================================================================
# Lee Wan Ching - homomorphic filtering, then Wiener denoising, then DWT
# detail enhancement. Source: pipeline_leewanching.ipynb
# =======================================================================

def homomorphic_filter(image, gamma_low, gamma_high, cutoff, c):
    """Frequency-domain illumination correction.

    Converts to grayscale, works in the log domain so that multiplicative
    illumination and additive reflectance separate into a sum, applies a
    high-frequency emphasis filter in the Fourier domain, then inverts
    both transforms. Reproduced exactly as written in
    pipeline_leewanching.ipynb.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) + 1
    log_img = np.log(gray)

    fft = np.fft.fft2(log_img)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = gray.shape
    crow = rows // 2
    ccol = cols // 2
    x, y = np.ogrid[:rows, :cols]
    distance = (x - crow) ** 2 + (y - ccol) ** 2

    high_pass = 1 - np.exp(-c * distance / (cutoff ** 2))
    H = gamma_low + (gamma_high - gamma_low) * high_pass

    result = fft_shift * H
    inverse = np.fft.ifftshift(result)
    output = np.fft.ifft2(inverse)
    output = np.exp(np.real(output))

    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    return output.astype(np.uint8)


def wiener_filter_image(image, kernel_size):
    """Wiener denoising, applied after the homomorphic filter above.

    Reproduced exactly as written in pipeline_leewanching.ipynb.
    """
    image = image.astype(np.float32)
    image = image + 1e-6
    result = wiener(image, (kernel_size, kernel_size))
    result = np.nan_to_num(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def dwt_detail_enhancement(image, wavelet, detail_gain):
    """Discrete wavelet detail boost, applied last in the chain.

    Decomposes into one approximation band (LL) and three detail bands
    (LH, HL, HH), scales the detail bands up, and reconstructs. Reproduced
    exactly as written in pipeline_leewanching.ipynb - the shape check
    that follows this step in leewanching_pipeline() below is what handles
    the size shift this can introduce, not this function itself.
    """
    coeffs = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs

    LH *= detail_gain
    HL *= detail_gain
    HH *= detail_gain

    result = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def leewanching_pipeline(image_bgr, gamma_low, gamma_high, cutoff, c,
                         kernel_size, wavelet, detail_gain):
    """The full three-stage pipeline, chained as in leewanching_pipeline()
    in pipeline_leewanching.ipynb, with the shape-safety step added.
    """
    homomorphic = homomorphic_filter(image_bgr, gamma_low, gamma_high, cutoff, c)
    denoised = wiener_filter_image(homomorphic, kernel_size)
    detailed = dwt_detail_enhancement(denoised, wavelet, detail_gain)
    return _match_original_shape(detailed, image_bgr.shape)


# The three parameter sets actually used to build her dataset, copied
# verbatim from the PARAMETERS dict in pipeline_leewanching.ipynb.
LEEWANCHING_TECHNIQUES = {
    "set1": dict(gamma_low=0.90, gamma_high=1.15, cutoff=60, c=1,
                kernel_size=3, wavelet="db2", detail_gain=1.45),
    "set2": dict(gamma_low=0.80, gamma_high=1.30, cutoff=40, c=1,
                kernel_size=5, wavelet="haar", detail_gain=1.70),
    "set3": dict(gamma_low=0.95, gamma_high=1.40, cutoff=80, c=1,
                kernel_size=7, wavelet="db4", detail_gain=2.00),
}


# =======================================================================
# Lim Sze Ping - CLAHE, then gamma correction, then Laplacian sharpening.
# Source: prepare_limszeping.ipynb, cell 10 (enhance_pcb)
# =======================================================================

def limszeping_pipeline(image_bgr, clahe_clip_limit, gamma, laplacian_weight):
    """The full three-stage pipeline, reproduced exactly as enhance_pcb()
    in prepare_limszeping.ipynb (cell 10 - the version that actually built
    her real dataset).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    gamma_table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)
    gamma_image = cv2.LUT(clahe_image, gamma_table)

    laplacian = cv2.Laplacian(gamma_image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.addWeighted(gamma_image, 1.0, laplacian_abs, laplacian_weight, 0)

    # CLAHE, the gamma LUT and addWeighted all preserve shape exactly, so
    # unlike the wavelet pipeline above there is nothing to correct here -
    # _match_original_shape is not needed, and is left out deliberately
    # rather than applied defensively where it cannot do anything.
    return sharpened


# The three parameter sets actually used to build her dataset, copied
# verbatim from the PARAMETER_SETS dict in prepare_limszeping.ipynb cell 10.
LIMSZEPING_TECHNIQUES = {
    "set1": dict(clahe_clip_limit=2.0, gamma=1.2, laplacian_weight=-0.7),
    "set2": dict(clahe_clip_limit=3.0, gamma=1.5, laplacian_weight=-1.0),
    "set3": dict(clahe_clip_limit=4.0, gamma=0.8, laplacian_weight=-0.5),
}


# =======================================================================
# Registry consumed by app.py
# =======================================================================
# One entry per teammate. "function" takes (image_bgr, **one parameter
# set) and returns a grayscale image; "sets" is that person's named
# parameter choices, in the same {name: {param: value}} shape used by
# core/preprocessing.py's TECHNIQUES table.

CONTRIBUTORS = {
    "Lee Wan Ching": {
        "function": leewanching_pipeline,
        "sets": LEEWANCHING_TECHNIQUES,
    },
    "Lim Sze Ping": {
        "function": limszeping_pipeline,
        "sets": LIMSZEPING_TECHNIQUES,
    },
}
