"""Synthetic PCB defect data for testing and demo.

Generates realistic mock data for:
- Defect detection (bounding boxes + class labels)
- Federated learning (per-factory Non-IID distributions)
- Feature search (DINOv2-like embeddings)
"""

import numpy as np
from typing import Optional


def generate_pcb_image(
    width: int = 640,
    height: int = 480,
    defect_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic PCB-like image.

    Creates a green PCB background with copper traces and optional defect overlay.
    """
    rng = np.random.RandomState(seed)

    # PCB substrate (dark green)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 1] = rng.randint(30, 60)  # Green channel

    # Copper traces (horizontal + vertical lines)
    for _ in range(rng.randint(5, 15)):
        if rng.random() > 0.5:
            y = rng.randint(0, height)
            x1, x2 = rng.randint(0, width), rng.randint(0, width)
            img[y-1:y+1, min(x1,x2):max(x1,x2)] = [180, 140, 50]  # Copper color
        else:
            x = rng.randint(0, width)
            y1, y2 = rng.randint(0, height), rng.randint(0, height)
            img[min(y1,y2):max(y1,y2), x-1:x+1] = [180, 140, 50]

    # Solder pads (circles)
    for _ in range(rng.randint(3, 8)):
        cx = rng.randint(max(5, width//4), max(6, width - width//4))
        cy = rng.randint(max(5, height//4), max(6, height - height//4))
        r = rng.randint(5, 15)
        y_grid, x_grid = np.ogrid[-cy:height-cy, -cx:width-cx]
        mask = x_grid**2 + y_grid**2 <= r**2
        img[mask] = [200, 160, 60]

    # Add defect overlay
    if defect_type:
        img = _add_defect(img, defect_type, rng)

    return img


def _add_defect(img: np.ndarray, defect_type: str, rng: np.random.RandomState) -> np.ndarray:
    """Add a synthetic defect to a PCB image."""
    h, w = img.shape[:2]
    cx, cy = rng.randint(w//4, 3*w//4), rng.randint(h//4, 3*h//4)

    if defect_type == "short":
        # Copper bridge between two traces
        y = cy
        img[y-2:y+2, cx-30:cx+30] = [220, 180, 60]

    elif defect_type == "open_circuit":
        # Gap in a trace
        y = cy
        img[y-1:y+2, cx-5:cx+5] = [20, 40, 20]  # Remove copper

    elif defect_type == "spurious_copper":
        # Random copper blob
        r = rng.randint(10, 25)
        y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x_grid**2 + y_grid**2 <= r**2
        img[mask] = [200, 160, 60]

    elif defect_type == "missing_hole":
        # Filled drill hole (should be empty)
        r = rng.randint(5, 12)
        y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x_grid**2 + y_grid**2 <= r**2
        img[mask] = [200, 160, 60]  # Should be dark

    elif defect_type == "spur":
        # Small copper protrusion
        for dy in range(-8, 8):
            dx = int(3 * np.sin(dy * 0.5))
            if 0 <= cy+dy < h and 0 <= cx+dx < w:
                img[cy+dy, cx+dx] = [200, 160, 60]

    elif defect_type == "mouse_bite":
        # Semicircular notch on trace edge
        r = rng.randint(5, 10)
        y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x_grid**2 + y_grid**2 <= r**2) & (x_grid >= 0)
        img[mask] = [20, 40, 20]

    return img


def generate_factory_data(
    factory_name: str,
    n_samples: int = 100,
    n_classes: int = 6,
    seed: Optional[int] = None,
) -> dict:
    """Generate Non-IID synthetic data for a factory.

    Returns dict with:
    - features: (n_samples, 768) float32 array
    - labels: (n_samples,) int array
    - class_distribution: dict of class -> count
    """
    rng = np.random.RandomState(seed)

    # Non-IID: each factory has dominant defect types
    class_weights = rng.dirichlet(np.ones(n_classes) * 0.5)
    labels = rng.choice(n_classes, size=n_samples, p=class_weights)

    # DINOv2-like features (768-dim, clustered by class)
    features = np.zeros((n_samples, 768), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        n_c = mask.sum()
        center = rng.randn(768).astype(np.float32) * 2
        features[mask] = center + rng.randn(n_c, 768).astype(np.float32) * 0.5

    class_distribution = {}
    for c in range(n_classes):
        class_distribution[c] = int((labels == c).sum())

    return {
        "factory": factory_name,
        "features": features,
        "labels": labels,
        "class_distribution": class_distribution,
    }


def generate_multi_factory_data(
    n_factories: int = 3,
    samples_per_factory: int = 100,
    seed: Optional[int] = None,
) -> list[dict]:
    """Generate data for multiple factories with different distributions."""
    factory_names = ["shenzhen_smt", "dongguan_pcb", "suzhou_hdi"]
    data = []
    for i in range(n_factories):
        name = factory_names[i % len(factory_names)]
        data.append(generate_factory_data(
            factory_name=name,
            n_samples=samples_per_factory,
            seed=seed + i if seed else None,
        ))
    return data
