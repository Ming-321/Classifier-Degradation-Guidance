import torch
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
import os
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    raise ImportError("Please install torchmetrics library: pip install torchmetrics")

from .base import BaseMetric


class FID(BaseMetric):
    """
    Fréchet Inception Distance (FID) evaluation metric implementation
    """

    def __init__(self, feature_dim: int = 2048, **kwargs):
        """
        Initialize FID evaluation metric

        Args:
            feature_dim: Inception feature dimension, default is 2048
            **kwargs: Other specific parameters
        """
        super().__init__(**kwargs)
        self.name = "FID"
        self.feature_dim = feature_dim

        # Set device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def calculate(self, generated_images_dir: str, ground_truth_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Calculate FID evaluation metric

        Args:
            generated_images_dir: Directory of generated images
            ground_truth_dir: Directory of ground truth images
            **kwargs: Other calculation parameters

        Returns:
            Dict[str, Any]: Calculation results, containing FID score
        """
        print(f"Calculate {self.name} metric...")

        # Initialize FID calculator
        fid = FrechetInceptionDistance(feature=self.feature_dim, normalize=True)
        fid = fid.to(self.device)

        # Get image paths
        gen_images_paths = self.get_image_paths(generated_images_dir)
        gt_images_paths = self.get_image_paths(ground_truth_dir)

        print(f"Found {len(gen_images_paths)} generated images and {len(gt_images_paths)} real images")

        if len(gen_images_paths) == 0 or len(gt_images_paths) == 0:
            return {"metric": self.name, "score": None, "error": "No images found"}

        # Batch process images
        batch_size = kwargs.get("batch_size", 32)

        # Process generated images
        print("Processing generated images...")
        self._process_images(fid, gen_images_paths, batch_size, is_real=False)

        # Process real images
        print("Processing real images...")
        self._process_images(fid, gt_images_paths, batch_size, is_real=True)

        # Calculate FID score
        fid_score = float(fid.compute().cpu().numpy())
        print(f"FID Score: {fid_score:.4f}")

        # Return results
        return {
            "metric": self.name,
            "score": fid_score,
            "num_generated_images": len(gen_images_paths),
            "num_real_images": len(gt_images_paths),
        }

    def _process_images(self, fid, image_paths: List[str], batch_size: int, is_real: bool) -> None:
        """
        Batch process images

        Args:
            fid: FID calculator
            image_paths: List of image paths
            batch_size: Batch size
            is_real: Whether it's real images
        """
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_resized = img.resize((299, 299))  # Inception-v3 input size
                    img_np = np.array(img_resized)
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # Convert to [C, H, W] format uint8 tensor
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                if is_real:
                    fid.update(batch_tensor, real=True)
                else:
                    fid.update(batch_tensor, real=False)
