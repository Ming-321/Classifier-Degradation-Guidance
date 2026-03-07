#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genai-Bench Analysis Program
Used to analyze the performance of different models on the Genai-Bench dataset, categorized by basic skills and advanced skills
"""

import json
import pandas as pd
import argparse
import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class GenAIBenchAnalyzer:
    def __init__(self, genai_bench_path: str = "evaluation/Genai-Bench/Genai-Bench.json"):
        """
        Initialize analyzer

        Args:
            genai_bench_path: Genai-Bench.json file path
        """
        self.genai_bench_path = genai_bench_path
        self.prompts_data = self.load_genai_bench_data()

        # Define skill categories
        self.basic_skills = ["Action Relation", "Attribute", "Scene", "Spatial Relation", "Part Relation"]
        self.advanced_skills = ["Counting", "Comparison", "Differentiation", "Negation", "Universal"]

    def load_genai_bench_data(self) -> List[Dict]:
        """Load Genai-Bench data"""
        try:
            with open(self.genai_bench_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} prompts")
            return data
        except Exception as e:
            raise FileNotFoundError(f"Cannot load Genai-Bench data: {e}")

    def load_detailed_scores(self, score_file_path: str) -> Dict[int, float]:
        """
        Load detailed score file

        Args:
            score_file_path: detailed_scores.csv file path

        Returns:
            Dictionary with prompt id as key and VQA score as value
        """
        try:
            df = pd.read_csv(score_file_path)
            scores = {}

            for _, row in df.iterrows():
                # Extract id from image filename
                image_name = row["image_name"]
                prompt_id = int(image_name.split(".")[0])

                # Parse VQAScore (may be string form of list)
                vqa_score = row["VQAScore"]
                if isinstance(vqa_score, str):
                    try:
                        score_list = ast.literal_eval(vqa_score)
                        if isinstance(score_list, list) and len(score_list) > 0:
                            score = float(score_list[0])
                        else:
                            score = float(vqa_score)
                    except:
                        score = float(vqa_score)
                else:
                    score = float(vqa_score)

                scores[prompt_id] = score

            print(f"Loaded {len(scores)} scores from {score_file_path}")
            return scores

        except Exception as e:
            print(f"Warning: Cannot load score file {score_file_path}: {e}")
            return {}

    def parse_skills(self, skills_str: str) -> List[str]:
        """Parse skills string, return skills list"""
        if not skills_str or skills_str.strip() == "":
            return []
        return [skill.strip() for skill in skills_str.split(",") if skill.strip()]

    def calculate_skill_scores(self, model_scores: Dict[str, Dict[int, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate average scores for each skill

        Args:
            model_scores: Mapping from model name to score dictionary

        Returns:
            (basic_skills_df, advanced_skills_df) two dataframes
        """
        # Initialize result dictionary (methods as rows)
        basic_results = {model_name: {} for model_name in model_scores.keys()}
        advanced_results = {model_name: {} for model_name in model_scores.keys()}

        # Calculate skill scores for each model
        for model_name, scores in model_scores.items():
            print(f"Processing model: {model_name}")

            # Basic skills statistics
            for skill in self.basic_skills:
                skill_scores = []
                for prompt in self.prompts_data:
                    prompt_id = prompt["id"]
                    if prompt_id in scores:
                        basic_skills = self.parse_skills(prompt["basic"])
                        if skill in basic_skills:
                            skill_scores.append(scores[prompt_id])

                if skill_scores:
                    basic_results[model_name][skill] = np.mean(skill_scores)
                    print(f"  Basic skill {skill}: {len(skill_scores)} samples, average score: {np.mean(skill_scores):.4f}")
                else:
                    basic_results[model_name][skill] = 0.0
                    print(f"  Basic skill {skill}: No samples found")

            # Advanced skills statistics
            for skill in self.advanced_skills:
                skill_scores = []
                for prompt in self.prompts_data:
                    prompt_id = prompt["id"]
                    if prompt_id in scores:
                        advanced_skills = self.parse_skills(prompt["advance"])
                        if skill in advanced_skills:
                            skill_scores.append(scores[prompt_id])

                if skill_scores:
                    advanced_results[model_name][skill] = np.mean(skill_scores)
                    print(f"  Advanced skill {skill}: {len(skill_scores)} samples, average score: {np.mean(skill_scores):.4f}")
                else:
                    advanced_results[model_name][skill] = 0.0
                    print(f"  Advanced skill {skill}: No samples found")

        # Convert to DataFrame (methods as rows, skills as columns)
        basic_df = pd.DataFrame.from_dict(basic_results, orient="index")
        advanced_df = pd.DataFrame.from_dict(advanced_results, orient="index")

        # Ensure columns are arranged according to defined skill order
        basic_df = basic_df[self.basic_skills]
        advanced_df = advanced_df[self.advanced_skills]

        # Add average column (average of each method across all skills)
        basic_df["Avg"] = basic_df.mean(axis=1)
        advanced_df["Avg"] = advanced_df.mean(axis=1)

        return basic_df, advanced_df

    def analyze_experiments(self, experiment_paths: List[str], output_dir: str = "./"):
        """
        Analyze results from multiple experiment paths

        Args:
            experiment_paths: List of experiment paths
            output_dir: Output directory
        """
        model_scores = {}

        # Collect scores from all models
        for exp_path in experiment_paths:
            path_obj = Path(exp_path)
            model_name = path_obj.name  # Use folder name as model name

            # Find evaluation/detailed_scores.csv file
            score_file = path_obj / "evaluation" / "detailed_scores.csv"
            if score_file.exists():
                scores = self.load_detailed_scores(str(score_file))
                if scores:
                    model_scores[model_name] = scores
                else:
                    print(f"Warning: No valid score data in {exp_path}")
            else:
                print(f"Warning: evaluation/detailed_scores.csv not found in {exp_path}")

        if not model_scores:
            print("Error: No valid score data found")
            return

        print(f"\nSuccessfully loaded data from {len(model_scores)} models")

        # Calculate skill scores
        basic_df, advanced_df = self.calculate_skill_scores(model_scores)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        basic_output = output_path / "basic_skills_analysis.csv"
        advanced_output = output_path / "advanced_skills_analysis.csv"

        # Format values, keep 5 decimal places
        basic_df_formatted = basic_df.round(5)
        advanced_df_formatted = advanced_df.round(5)

        basic_df_formatted.to_csv(basic_output, index_label="Method")
        advanced_df_formatted.to_csv(advanced_output, index_label="Method")

        print(f"\nResults saved to:")
        print(f"Basic skills analysis: {basic_output}")
        print(f"Advanced skills analysis: {advanced_output}")

        # Print summary
        print("\n=== Basic Skills Analysis Summary ===")
        print(basic_df_formatted)

        print("\n=== Advanced Skills Analysis Summary ===")
        print(advanced_df_formatted)


def main():
    parser = argparse.ArgumentParser(description="Genai-Bench Data Analysis Tool")
    parser.add_argument("paths", nargs="+", help="One or more experiment paths")
    parser.add_argument(
        "--genai-bench", default="evaluation/Genai-Bench/Genai-Bench.json", help="Genai-Bench.json file path"
    )
    parser.add_argument("--output-dir", default="./analysis_results", help="Output directory")

    args = parser.parse_args()

    # Check if paths exist
    valid_paths = []
    for path in args.paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: Path does not exist {path}")

    if not valid_paths:
        print("Error: No valid paths found")
        return

    # Initialize analyzer
    try:
        analyzer = GenAIBenchAnalyzer(args.genai_bench)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Execute analysis
    analyzer.analyze_experiments(valid_paths, args.output_dir)


if __name__ == "__main__":
    main()
