"""Image preprocessing utilities.

The quality of the OCR output is heavily influenced by the input
images.  This module defines a few simple preprocessing pipelines
implemented using Pillow and OpenCV.  Each pipeline takes a PIL image
and returns a processed PIL image.  The default pipeline converts the
image to grayscale and applies unsharp masking and autocontrast; the
binary pipeline performs Otsu thresholding; and the OpenCV pipeline
implements CLAHE and adaptive thresholding with slight deskewing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


def pil_gray(img: Image.Image, crop_pct: float = 0.0) -> Image.Image:
    """Convert to grayscale and enhance contrast.

    The ``crop_pct`` parameter can be used to remove a small margin
    around the image prior to processing.
    """
    if crop_pct > 0:
        w, h = img.size
        dx = int(w * crop_pct)
        dy = int(h * crop_pct)
        img = img.crop((dx, dy, w - dx, h - dy))
    gray = img.convert("L")
    # apply unsharp mask to sharpen text
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    # auto contrast to normalise histogram
    ac = ImageOps.autocontrast(sharp)
    return ac


def pil_bin(img: Image.Image, crop_pct: float = 0.0) -> Image.Image:
    """Binarise the image using Otsu's threshold after grayscale.

    Suitable for pages with clean backgrounds.  Adds a margin crop
    similar to :func:`pil_gray`.
    """
    gray = pil_gray(img, crop_pct)
    arr = np.array(gray)
    # compute Otsu threshold using OpenCV
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def opencv(img: Image.Image, crop_pct: float = 0.0) -> Image.Image:
    """Use OpenCV to deskew and threshold the image.

    This pipeline converts the input to grayscale, applies Contrast
    Limited Adaptive Histogram Equalisation (CLAHE), then performs
    adaptive thresholding.  A small deskew (within ±3°) is applied
    using the minimum area rectangle from the contours.
    """
    if crop_pct > 0:
        w, h = img.size
        dx = int(w * crop_pct)
        dy = int(h * crop_pct)
        img = img.crop((dx, dy, w - dx, h - dy))
    arr = np.array(img.convert("L"))
    # apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    # adaptive thresholding
    thresh = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)
    # deskew by computing the minimum area rect of contours
    coords = np.column_stack(np.where(thresh < 255))
    angle = 0.0
    if len(coords) > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
    # rotate to correct skew
    if abs(angle) > 0.1:
        (h, w) = thresh.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(thresh)


def preprocess(img: Image.Image, profile: str, crop_pct: float = 0.0) -> Image.Image:
    """Dispatch to the appropriate preprocessing pipeline.

    Parameters
    ----------
    img:
        The input image to process.
    profile:
        One of ``"pil_gray"``, ``"pil_bin"`` or ``"opencv"``.
    crop_pct:
        Fractional margin to remove from all sides before processing.
    """
    if profile == "pil_gray":
        return pil_gray(img, crop_pct)
    if profile == "pil_bin":
        return pil_bin(img, crop_pct)
    if profile == "opencv":
        return opencv(img, crop_pct)
    raise ValueError(f"Unknown preprocessing profile: {profile}")