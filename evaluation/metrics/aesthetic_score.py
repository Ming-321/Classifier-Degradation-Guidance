import torch
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
import os
from tqdm import tqdm
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip


from .base import BaseMetric


class AestheticScore(BaseMetric):
    """
    Aesthetic Score v2 evaluation metric implementation
    """

    def __init__(self, **kwargs):
        """
        Initialize Aesthetic Score evaluation metric

        Args:
            **kwargs: 
                - encoder_model: SigLIP model path (default: google/siglip-so400m-patch14-384)
                - predictor_path: predictor weight path (default: auto-download from GitHub)
                - device: device (default: cuda)
                - batch_size: batch size
        """
        super().__init__(**kwargs)
        self.name = "AestheticScoreV2"

        # Set device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Get local path config (supports reading from metrics.json)
        encoder_model = kwargs.get("encoder_model", "google/siglip-so400m-patch14-384")
        predictor_path = kwargs.get("predictor_path", None)

        # Load model
        print("Loading Aesthetic Score v2 model...")
        print(f"  Encoder model: {encoder_model}")
        print(f"  Predictor path: {predictor_path if predictor_path else 'will download from GitHub'}")
        model_result = convert_v2_5_from_siglip(
            predictor_name_or_path=predictor_path,
            encoder_model_name=encoder_model,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model, self.processor = model_result
        self.model = self.model.to(self.device, dtype=self.dtype)  # type: ignore
        print("Aesthetic Score v2 model loaded successfully")

    def calculate(self, generated_images_dir: str, ground_truth_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate Aesthetic Score

        Args:
            generated_images_dir: Directory of generated images
            ground_truth_dir: Directory of ground truth images（not used）
            **kwargs: Other calculation parameters

        Returns:
            Dict[str, Any]: Calculation results
        """
        print(f"Calculate {self.name} metric...")

        gen_images_paths = self.get_image_paths(generated_images_dir)

        if len(gen_images_paths) == 0:
            return {"metric": self.name, "score": None, "error": "No generated images found"}

        print(f"Found {len(gen_images_paths)} generated images")

        batch_size = kwargs.get("batch_size", 16)
        all_scores = []

        print(f"Calculating Aesthetic scores for {len(gen_images_paths)} images...")
        for i in tqdm(range(0, len(gen_images_paths), batch_size)):
            batch_paths = gen_images_paths[i : i + batch_size]
            batch_scores = self._compute_batch_aesthetic_score(batch_paths)
            all_scores.extend(batch_scores)

        mean_aesthetic_score = float(np.mean(all_scores))
        print(f"Average Aesthetic Score: {mean_aesthetic_score:.4f}")

        return {
            "metric": self.name,
            "score": mean_aesthetic_score,
            "individual_scores": dict(zip([os.path.basename(p) for p in gen_images_paths], all_scores)),
            "num_evaluated_images": len(gen_images_paths),
        }

    def _compute_batch_aesthetic_score(self, image_paths: List[str]) -> List[float]:
        """
        Calculate Aesthetic Score for a batch of images

        Args:
            image_paths: List of image paths

        Returns:
            List[float]: List of scores
        """
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error processing image {path}: {e}")

        if not images:
            return []

        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            scores = self.model(pixel_values).logits.squeeze().float().cpu().numpy()

        return scores.tolist()
