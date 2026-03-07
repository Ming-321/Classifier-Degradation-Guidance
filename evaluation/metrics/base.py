from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import json


class BaseMetric(ABC):
    """
    Base class for evaluation metrics. All evaluation metric implementations must inherit from this class and implement its abstract methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize evaluation metric

        Args:
            **kwargs: Metric-specific parameters
        """
        self.name = self.__class__.__name__

    @abstractmethod
    def calculate(self, generated_images_dir: str, ground_truth_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Calculate evaluation metric

        Args:
            generated_images_dir: Directory of generated images
            ground_truth_dir: Directory of ground truth images
            **kwargs: Other calculation parameters

        Returns:
            Dict[str, Any]: Calculation results, including scores, metric names, etc.
        """
        pass

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save evaluation results to file

        Args:
            results: Evaluation results
            output_file: Output file path
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Evaluation results saved to {output_file}")

    def get_image_paths(self, directory: str, extensions: List[str] = [".jpg", ".jpeg", ".png"]) -> List[str]:
        """
        Get all image paths in directory

        Args:
            directory: Image directory
            extensions: List of image extensions

        Returns:
            List[str]: List of image paths
        """
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def __str__(self) -> str:
        """
        Return metric name
        """
        return self.name
