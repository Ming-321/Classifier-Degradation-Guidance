#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import importlib
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Use environment check tool to set up path
from .utils.env_check import setup_project_path

setup_project_path()

# Now can use absolute imports
from evaluation.metrics.base import BaseMetric


def get_metric_class(metric_name: str):
    """
    Get the corresponding metric class based on metric name.
    Fixed metric names, case-sensitive or underscores will cause errors.

    Args:
        metric_name: Metric name, must match exactly

    Returns:
        Corresponding metric class
    """
    # Fixed metric name mapping
    VALID_METRICS = {
        "AestheticScore": "aesthetic_score",
        "CLIPScore": "clip_score",
        "FID": "fid",
        "VQAScore": "vqa_score",
    }

    if metric_name not in VALID_METRICS:
        raise ValueError(f"Metric name must exactly match one of: {list(VALID_METRICS.keys())}")

    try:
        module_name = VALID_METRICS[metric_name]
        module = importlib.import_module(f"evaluation.metrics.{module_name}")

        # Find class that inherits from BaseMetric
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseMetric) and attr != BaseMetric:
                return attr

        raise ImportError(f"Cannot find class inheriting from BaseMetric in module 'evaluation.metrics.{module_name}'")
    except ImportError as e:
        print(f"Error: {e}")
        raise


def load_prompts_mapping(generation_dir: str) -> Dict[str, str]:
    """
    Load mapping relationship between generated images and prompts from prompts.json.

    Args:
        generation_dir: Directory of generated images, should contain prompts.json file

    Returns:
        Dict[str, str]: Mapping from image path to prompt
    """
    prompts_json_path = os.path.join(generation_dir, "prompts.json")

    if not os.path.exists(prompts_json_path):
        print(f"Warning: Cannot find prompts.json file at {prompts_json_path}")
        return {}

    with open(prompts_json_path, "r", encoding="utf-8") as f:
        try:
            prompt_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse prompts.json file")
            return {}

    mapping = {}
    for item in prompt_data:
        if "image_path" in item and "prompt" in item:
            # Construct full image path
            if item["image_path"].startswith("/"):
                # Absolute path
                image_path = item["image_path"]
            else:
                # Relative path, relative to generation_dir
                image_path = os.path.join(generation_dir, item["image_path"])

            # Check if file exists
            if os.path.exists(image_path):
                mapping[image_path] = item["prompt"]
            else:
                print(f"Warning: Image in mapping file does not exist: {image_path}")

    print(f"Loaded {len(mapping)} image-prompt pairs from prompts.json")
    return mapping


def save_results(
    results: Dict[str, Any], output_file: str, detailed_scores: Optional[Dict[str, Dict[str, float]]] = None
) -> None:
    """
    Save evaluation results.

    Args:
        results: Evaluation results dictionary
        output_file: Output file path
        detailed_scores: Detailed scores dictionary, format: {metric_name: {image_name: score}}
    """
    # Prepare data to save
    result_data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"Evaluation results saved to: {output_file}")

    # Save detailed scores to CSV file
    if detailed_scores:
        csv_file = os.path.join(output_dir, "detailed_scores.csv")

        # Prepare DataFrame data
        df_data = []

        # Get all image names
        all_image_names = set()
        for metric_scores in detailed_scores.values():
            all_image_names.update(metric_scores.keys())

        # Create a row for each image
        for image_name in sorted(all_image_names):
            row = {"image_name": image_name}
            for metric_name, metric_scores in detailed_scores.items():
                row[metric_name] = metric_scores.get(image_name, None)
            df_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"Detailed scores saved to: {csv_file}")


def main():
    # Record start time
    start_time = time.time()

    # Parse command line arguments - simplified to only need two parameters
    parser = argparse.ArgumentParser(description="Simplified image generation evaluation script")

    parser.add_argument("--generated_path", type=str, required=True, help="Directory path of generated images")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Metrics to calculate, comma-separated, e.g. AestheticScore,CLIPScore",
    )
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")

    args = parser.parse_args()

    # Ensure generated images path exists
    if not os.path.exists(args.generated_path):
        raise FileNotFoundError(f"Generated images directory does not exist: {args.generated_path}")

    # Use unified configuration management from configs
    try:
        from configs.utils import get_evaluation_parameters, prepare_metric_kwargs

        # Get evaluation parameters
        eval_params = get_evaluation_parameters("coco2017")
        gt_path = eval_params["gt_path"]
        metrics_config = eval_params["metrics_config"]

    except Exception as e:
        raise RuntimeError(f"Failed to load evaluation config from configs: {e}")

    # Parse metrics to calculate
    metric_names = [name.strip() for name in args.metrics.split(",")]

    # Set result file path - fixed save to generated_path/evaluation/ directory
    evaluation_dir = os.path.join(args.generated_path, "evaluation")
    result_file = os.path.join(evaluation_dir, "metrics_results.json")

    # Load prompts mapping (for metrics that need prompts)
    prompts_mapping = load_prompts_mapping(args.generated_path)

    # Calculate metrics
    results = {}
    detailed_scores = {}  # Store detailed scores for each image

    for metric_name in metric_names:
        print(f"\nCalculating metric: {metric_name}")

        try:
            # Get metric class
            metric_class = get_metric_class(metric_name)

            # Get metric configuration from configs
            metric_config = metrics_config.get(metric_name)

            if not metric_config:
                raise ValueError(f"Metric {metric_name} not configured in configs")

            # Use unified parameter preparation method from configs
            metric_kwargs = prepare_metric_kwargs(metric_name, metric_config, args.batch_size)

            # Create metric instance
            metric = metric_class(**metric_kwargs)

            # Calculate metric - use calculate method instead of compute_score
            calculate_kwargs = {}

            # Add prompts_mapping parameter for metrics that need prompts
            if metric_name in ["CLIPScore", "VQAScore"] and prompts_mapping:
                calculate_kwargs["prompts_mapping"] = prompts_mapping

            # Call calculate method
            result = metric.calculate(args.generated_path, gt_path, **calculate_kwargs)

            # Extract score
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
            else:
                score = result

            # Collect detailed scores and calculate variance (only for specified three metrics)
            result_metric_name = result.get("metric", metric_name) if isinstance(result, dict) else metric_name
            if result_metric_name in ["CLIPScore", "AestheticScoreV2", "VQAScore"] and isinstance(result, dict):
                individual_scores = result.get("individual_scores", {})
                if individual_scores:
                    # Save detailed scores
                    detailed_scores[metric_name] = individual_scores

                    # Calculate variance
                    scores_values = list(individual_scores.values())
                    if scores_values:
                        variance = float(np.var(scores_values))
                        std_dev = float(np.std(scores_values))

                        # Update results with mean, variance and standard deviation
                        results[metric_name] = {
                            "mean": score,
                            "variance": variance,
                            "std_dev": std_dev,
                            "count": len(scores_values),
                        }
                        print(f"{metric_name}: mean={score:.4f}, variance={variance:.4f}, std_dev={std_dev:.4f}")
                    else:
                        results[metric_name] = score
                        print(f"{metric_name}: {score}")
                else:
                    results[metric_name] = score
                    print(f"{metric_name}: {score}")
            else:
                results[metric_name] = score
                print(f"{metric_name}: {score}")

        except Exception as e:
            print(f"Error calculating metric {metric_name}: {e}")
            results[metric_name] = None

    # Save results
    save_results(results, result_file, detailed_scores)

    # Print summary
    total_time = time.time() - start_time
    print(f"\nEvaluation completed! Total time: {total_time:.2f} seconds")
    print(f"Results summary:")
    for metric_name, result in results.items():
        if result is not None:
            if isinstance(result, dict) and "mean" in result:
                # For metrics containing statistical information
                print(f"  {metric_name}:")
                print(f"    Mean: {result['mean']:.4f}")
                print(f"    Variance: {result['variance']:.4f}")
                print(f"    Standard deviation: {result['std_dev']:.4f}")
                print(f"    Sample count: {result['count']}")
            else:
                # For regular scores
                print(f"  {metric_name}: {result}")
        else:
            print(f"  {metric_name}: Calculation failed")

    # Show usage example
    print(f"\nNext usage example:")
    print(
        f"python evaluation/eval.py --generated_path {args.generated_path} --metrics AestheticScore,FID,CLIPScore --batch_size 250"
    )


if __name__ == "__main__":
    main()
