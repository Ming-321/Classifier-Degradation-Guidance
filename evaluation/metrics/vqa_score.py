#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image
import os
from tqdm import tqdm

try:
    import t2v_metrics
except ImportError:
    t2v_metrics = None

from .base import BaseMetric


class VQAScore(BaseMetric):
    """
    VQA Score evaluation metric implementation using CLIP-FlanT5-XXL model
    Based on t2v_metrics library
    """

    def __init__(self, model_name: str = "clip-flant5-xxl", **kwargs):
        """
        Initialize VQAScore evaluation metric

        Args:
            model_name: VQA model name, default is "clip-flant5-xxl"
            **kwargs: Other specific parameters
        """
        super().__init__(**kwargs)
        self.name = "VQAScore"
        self.model_name = model_name

        if t2v_metrics is None:
            raise ImportError(
                "t2v_metrics is required for VQAScore. Install with: pip install t2v_metrics"
            )

        # Set device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize VQAScore model
        print(f"Loading VQAScore model: {self.model_name} on {self.device}...")
        try:
            self.vqa_model = t2v_metrics.VQAScore(model=self.model_name, device=self.device)
            print("VQAScore model loaded successfully")
        except Exception as e:
            print(f"Error loading VQAScore model: {e}")
            raise

    def calculate(
        self, generated_images_dir: str, ground_truth_dir: str = None, **kwargs  # VQAScore does not need real images
    ) -> Dict[str, Any]:
        """
        Calculate VQA Score evaluation metric

        Args:
            generated_images_dir: Directory of generated images
            ground_truth_dir: Directory of ground truth images（for VQAScore not used）
            **kwargs: Other calculation parameters，must include prompts_mapping parameter

        Returns:
            Dict[str, Any]: Calculation results, containing VQAScore
        """
        print(f"Calculate {self.name} metric...")

        # VQAScore requires prompts
        prompts_mapping = kwargs.get("prompts_mapping")
        if prompts_mapping is None:
            raise ValueError("VQAScore metric requires prompts_mapping (loaded from prompts.json)")

        # Get generated image paths
        gen_images_paths = self.get_image_paths(generated_images_dir)

        if len(gen_images_paths) == 0:
            return {"metric": self.name, "score": None, "error": "No generated images found"}

        print(f"Found {len(gen_images_paths)} generated images")

        # Build image-prompt pairs
        image_prompt_pairs = []
        for img_path in gen_images_paths:
            # Use normalized path
            normalized_img_path = os.path.normpath(img_path)
            prompt = prompts_mapping.get(normalized_img_path)

            if prompt:
                image_prompt_pairs.append((img_path, prompt))
            else:
                # Try to match by filename
                base_filename = os.path.basename(normalized_img_path)
                found_prompt = None
                for key, value in prompts_mapping.items():
                    if os.path.basename(key) == base_filename:
                        found_prompt = value
                        break

                if found_prompt:
                    image_prompt_pairs.append((img_path, found_prompt))
                else:
                    print(f"Warning: Cannot find prompt for image '{base_filename}' in prompts.json")

        if not image_prompt_pairs:
            return {"metric": self.name, "score": None, "error": "Cannot find valid image-prompt pairs"}

        print(f"Calculating VQA scores for {len(image_prompt_pairs)} image-prompt pairs...")

        # Batch processing
        all_scores = []

        # Calculate scores in batches
        for image, text in tqdm(image_prompt_pairs):
            batch_scores = self.vqa_model(
                images=image,
                texts=text,
            )

            # Convert tensor to list
            if isinstance(batch_scores, torch.Tensor):
                batch_scores = batch_scores.cpu().numpy().tolist()
            elif isinstance(batch_scores, np.ndarray):
                batch_scores = batch_scores.tolist()

            all_scores.extend(batch_scores)

        # Calculate average score
        if all_scores:
            mean_vqa_score = float(np.mean(all_scores))
            print(f"Average VQA score: {mean_vqa_score:.4f}")
        else:
            mean_vqa_score = 0.0
            print("Warning: No VQA scores were successfully calculated")

        # Return results
        return {
            "metric": self.name,
            "score": mean_vqa_score,
            "model_name": self.model_name,
            "individual_scores": dict(zip([os.path.basename(pair[0]) for pair in image_prompt_pairs], all_scores)),
            "num_evaluated_images": len(image_prompt_pairs),
        }

    def set_templates(self, question_template: str, answer_template: str):
        """
        Set custom question-answer templates

        Args:
            question_template: Question template，{} will be replaced by prompt
            answer_template: Answer template
        """
        self.default_question_template = question_template
        self.default_answer_template = answer_template
        print(f"Custom templates set:")
        print(f"Question template: {question_template}")
        print(f"Answer template: {answer_template}")


