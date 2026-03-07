import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import os
import re
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


from .base import BaseMetric


class CLIPScore(BaseMetric):
    """
    CLIP Score evaluation metric implementation
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", **kwargs):
        """
        Initialize CLIP Score evaluation metric

        Args:
            model_name: CLIP model name
            **kwargs: Other specific parameters
        """
        super().__init__(**kwargs)
        self.name = "CLIPScore"
        self.model_name = model_name

        # Set device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model
        print(f"Load CLIP model: {self.model_name}...")
        self.model = CLIPModel.from_pretrained(self.model_name)
        processor_result = CLIPProcessor.from_pretrained(self.model_name)
        self.processor = processor_result if not isinstance(processor_result, tuple) else processor_result[0]
        self.model.to(self.device)
        print("CLIP model loaded successfully")

    def calculate(
        self,
        generated_images_dir: str,
        ground_truth_dir: Optional[str] = None,  # Note: CLIP Score does not need real images
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate CLIP Score evaluation metric

        Args:
            generated_images_dir: Directory of generated images
            ground_truth_dir: Directory of ground truth images（not used for CLIP Score）
            **kwargs: Other calculation parameters，must include prompts_mapping parameter

        Returns:
            Dict[str, Any]: Calculation results, containing CLIP Score
        """
        print(f"Calculate {self.name} metric...")

        # CLIP Score requires prompts, obtained from filename or mapping
        prompts_mapping = kwargs.get("prompts_mapping", None)

        # Get generated image paths
        gen_images_paths = self.get_image_paths(generated_images_dir)

        if len(gen_images_paths) == 0:
            return {"metric": self.name, "score": None, "error": "No generated images found"}

        print(f"Found {len(gen_images_paths)} generated images")

        # Batch process images
        batch_size = kwargs.get("batch_size", 16)

        # Calculate CLIP score for each image
        all_scores = []
        image_prompt_pairs = []

        # Build image-prompt pairs
        if prompts_mapping:
            # Use provided mapping
            for image_path, prompt in prompts_mapping.items():
                if image_path in gen_images_paths:
                    image_prompt_pairs.append((image_path, prompt))
        else:
            # No prompt mapping, try to infer from filename or use default description
            for image_path in gen_images_paths:
                # Can try to extract description from filename, or use generic description
                filename = os.path.basename(image_path)
                # Simple default description
                default_prompt = "an image"
                image_prompt_pairs.append((image_path, default_prompt))

        if not image_prompt_pairs:
            return {"metric": self.name, "score": None, "error": "Cannot find valid image-prompt pairs"}

        print(f"Calculating CLIP scores for {len(image_prompt_pairs)} image-prompt pairs...")

        # Batch calculate scores
        for i in tqdm(range(0, len(image_prompt_pairs), batch_size)):
            batch = image_prompt_pairs[i : i + batch_size]
            batch_paths = [pair[0] for pair in batch]
            batch_prompts = [pair[1] for pair in batch]

            # Calculate batch scores
            batch_scores = self._compute_batch_clip_score(batch_paths, batch_prompts)
            all_scores.extend(batch_scores)

        # Calculate average score
        mean_clip_score = float(np.mean(all_scores))
        print(f"Average CLIP score: {mean_clip_score:.4f}")

        # Return results
        return {
            "metric": self.name,
            "score": mean_clip_score,
            "individual_scores": dict(zip([os.path.basename(pair[0]) for pair in image_prompt_pairs], all_scores)),
            "num_evaluated_images": len(image_prompt_pairs),
        }

    def _compute_batch_clip_score(self, image_paths: List[str], prompts: List[str]) -> List[float]:
        """
        Calculate CLIP scores for a batch of images

        Args:
            image_paths: List of image paths
            prompts: Corresponding prompt list

        Returns:
            List[float]: List of scores
        """
        # Load images
        images = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error processing image {path}: {e}")

        if not images:
            return []

        # Get valid prompts
        valid_prompts = [prompts[i] for i in valid_indices]

        # Prepare inputs
        inputs = self.processor(
            text=valid_prompts, images=images, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Calculate features
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get text and image features
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize features
            image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

            # Calculate similarity
            similarity = torch.diagonal(torch.matmul(image_embeds, text_embeds.t())).cpu().numpy()

            # Convert to percentage scores
            scores = similarity * 100.0

        return scores.tolist()
