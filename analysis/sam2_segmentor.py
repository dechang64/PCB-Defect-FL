"""
analysis/sam2_segmentor.py — SAM2 Pixel-Level Defect Segmentation
==================================================================

Adapted from organoid-fl's SAM2 integration for industrial defect detection.

Pipeline:
1. YOLO detects defects → bounding boxes (prompts)
2. SAM2 uses boxes as prompts → pixel-level masks
3. Extract morphology metrics from masks

Morphology metrics:
- Area (pixel count)
- Perimeter (boundary length)
- Circularity (4π·area/perimeter², 1 = perfect circle)
- Solidity (area/convex_area, measures concavity)
- Aspect ratio (major/minor axis from ellipse fit)
- Eccentricity (0 = circle, 1 = line)

Falls back to contour-based segmentation when SAM2 unavailable.
Streamlit Cloud compatible.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


@dataclass
class DefectMask:
    """Pixel-level defect segmentation result."""
    mask: np.ndarray          # binary mask (H, W)
    bbox: List[float]         # [x1, y1, x2, y2] from YOLO
    defect_type: str
    confidence: float
    area: int = 0             # pixel count
    perimeter: float = 0.0
    circularity: float = 0.0  # 1.0 = perfect circle
    solidity: float = 0.0     # 1.0 = convex
    aspect_ratio: float = 1.0
    eccentricity: float = 0.0

    def __post_init__(self):
        """Compute morphology metrics from mask."""
        if self.mask is not None and self.mask.any():
            self._compute_morphology()

    def _compute_morphology(self):
        """Compute morphology metrics from binary mask."""
        mask = self.mask.astype(np.uint8)

        # Area
        self.area = int(np.sum(mask))

        # Perimeter (boundary pixel count)
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask).astype(np.uint8)
        boundary = mask - eroded
        self.perimeter = float(np.sum(boundary))

        # Circularity
        if self.perimeter > 0:
            self.circularity = min(4.0 * np.pi * self.area / (self.perimeter ** 2), 1.0)
        else:
            self.circularity = 0.0

        # Convex hull for solidity
        try:
            from scipy.spatial import ConvexHull
            ys, xs = np.where(mask > 0)
            if len(xs) > 3:
                points = np.column_stack([xs, ys])
                hull = ConvexHull(points)
                convex_area = hull.volume  # 2D: volume = area
                self.solidity = self.area / max(convex_area, 1)
            else:
                self.solidity = 1.0
        except Exception:
            self.solidity = 1.0

        # Aspect ratio and eccentricity from bounding box
        h = self.bbox[3] - self.bbox[1]
        w = self.bbox[2] - self.bbox[0]
        if h > 0 and w > 0:
            self.aspect_ratio = max(w, h) / min(w, h)
            a = max(w, h) / 2
            b = min(w, h) / 2
            self.eccentricity = np.sqrt(1 - (b / a) ** 2) if a > 0 else 0.0


@dataclass
class SegmentationResult:
    """Complete segmentation result for one image."""
    image_size: Tuple[int, int]  # (H, W)
    masks: List[DefectMask]
    total_defect_area: int = 0
    defect_area_ratio: float = 0.0
    method: str = "contour"  # "sam2" or "contour"

    def __post_init__(self):
        self.total_defect_area = sum(m.area for m in self.masks)
        h, w = self.image_size
        self.defect_area_ratio = self.total_defect_area / max(h * w, 1)


class DefectSAM2Segmentor:
    """SAM2-based pixel-level defect segmentation.

    Falls back to contour-based segmentation when SAM2 is unavailable.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.sam_predictor = None
        self.method = "contour"  # Default fallback

        if HAS_SAM:
            try:
                self._load_sam()
                self.method = "sam2"
            except Exception:
                self.method = "contour"

    def _load_sam(self):
        """Load SAM2 model."""
        # Try to load SAM2 checkpoint
        checkpoint_paths = [
            "checkpoints/sam2_hiera_large.pt",
            "checkpoints/sam2_hiera_base_plus.pt",
            "checkpoints/sam2_hiera_small.pt",
            "checkpoints/sam2_hiera_tiny.pt",
        ]
        import os
        for path in checkpoint_paths:
            if os.path.exists(path):
                sam = sam_model_registry["sam2_hiera_l"](checkpoint=path)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
                self.method = "sam2"
                return

    def segment(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> SegmentationResult:
        """Segment defects at pixel level.

        Args:
            image: (H, W, 3) RGB image.
            detections: List of {"bbox": [x1,y1,x2,y2], "class": str, "confidence": float}.

        Returns:
            SegmentationResult with pixel-level masks and morphology metrics.
        """
        h, w = image.shape[:2]
        masks = []

        if self.method == "sam2" and self.sam_predictor is not None:
            masks = self._segment_sam2(image, detections)
        else:
            masks = self._segment_contour(image, detections)

        return SegmentationResult(
            image_size=(h, w),
            masks=masks,
            method=self.method,
        )

    def _segment_sam2(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[DefectMask]:
        """SAM2-based segmentation."""
        self.sam_predictor.set_image(image)
        masks = []

        for det in detections:
            bbox = det.get("bbox", [0, 0, 100, 100])
            box = np.array(bbox)  # [x1, y1, x2, y2]

            mask_pred, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True,
            )

            # Use highest confidence mask
            best_idx = np.argmax(scores)
            mask = mask_pred[best_idx]

            masks.append(DefectMask(
                mask=mask,
                bbox=bbox,
                defect_type=det.get("class", "unknown"),
                confidence=det.get("confidence", 0.0),
            ))

        return masks

    def _segment_contour(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[DefectMask]:
        """Contour-based fallback segmentation.

        Uses Otsu thresholding within each bounding box to find defect pixels.
        """
        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None

        masks = []
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        for det in detections:
            bbox = det.get("bbox", [0, 0, 100, 100])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)

            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Otsu threshold within ROI
            threshold = np.mean(roi) - np.std(roi)
            binary = (roi < threshold).astype(np.uint8)

            # Clean up with morphological operations
            if ndimage is not None:
                binary = ndimage.binary_opening(binary, iterations=1).astype(np.uint8)
                binary = ndimage.binary_closing(binary, iterations=1).astype(np.uint8)

            # Place back in full image
            full_mask = np.zeros(gray.shape, dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = binary

            masks.append(DefectMask(
                mask=full_mask,
                bbox=bbox,
                defect_type=det.get("class", "unknown"),
                confidence=det.get("confidence", 0.0),
            ))

        return masks

    @staticmethod
    def compare_bbox_vs_pixel(
        bbox_mAP: float,
        pixel_mAP: float,
        defect_type: str,
    ) -> Dict[str, Any]:
        """Compare bounding box vs pixel-level evaluation.

        Args:
            bbox_mAP: mAP@0.5 from bounding box detection.
            pixel_mAP: mAP@0.5 from pixel-level segmentation.
            defect_type: Defect class being compared.

        Returns:
            Comparison analysis.
        """
        gap = pixel_mAP - bbox_mAP
        if gap < -0.05:
            assessment = (f"Pixel-level mAP is significantly lower ({gap:.3f}). "
                          f"Bounding boxes overestimate detection quality for '{defect_type}'. "
                          f"Consider: (1) pixel-level training, (2) tighter annotation, "
                          f"(3) SAM2-based post-processing.")
        elif gap < 0:
            assessment = (f"Pixel-level mAP is slightly lower ({gap:.3f}). "
                          f"Reasonable bbox-to-pixel gap for '{defect_type}'.")
        else:
            assessment = (f"Pixel-level mAP is equal or higher. "
                          f"'{defect_type}' has well-defined boundaries.")

        return {
            "defect_type": defect_type,
            "bbox_mAP": bbox_mAP,
            "pixel_mAP": pixel_mAP,
            "gap": gap,
            "assessment": assessment,
        }
