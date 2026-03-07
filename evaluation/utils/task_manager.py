#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

# Use environment check tool to set up path
from .env_check import setup_project_path

setup_project_path()

# Now can use absolute imports
from evaluation.core.model_interface import create_model_runner
from evaluation.core.paths import paths
from evaluation.utils.filename_utils import sanitize_filename


class TaskManager:
    """Task manager, handles GPU allocation and checkpoint resumption"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        self.metadata_file = self.output_dir / "metadata.json"

    def run_distributed_generation(
        self,
        model_type: str,
        model_variant: str,
        config_path: str,
        prompt_data: List[Tuple[str, Optional[str]]],
        gpu_ids: List[int],
        resume: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Run distributed generation on multiple GPUs

        Args:
            model_type: Model type
            model_variant: Model variant
            config_path: Configuration file path
            prompt_data: Prompt data
            gpu_ids: List of GPU IDs
            resume: Whether to support checkpoint resumption
            seed: Random seed for image generation
        """
        print(f"Starting distributed generation task")
        print(f"Model: {model_type}/{model_variant}")
        print(f"Using GPUs: {gpu_ids}")
        print(f"Total prompt count: {len(prompt_data)}")

        # If checkpoint resumption is supported, check completed tasks
        remaining_prompts = []
        if resume:
            remaining_prompts = self._filter_completed_tasks(prompt_data)
        else:
            remaining_prompts = prompt_data

        if not remaining_prompts:
            print("All tasks completed!")
            return

        print(f"Remaining task count: {len(remaining_prompts)}")

        # Allocate tasks to GPUs
        gpu_tasks = self._allocate_tasks_to_gpus(remaining_prompts, gpu_ids)

        # Start GPU tasks
        processes = []
        temp_files = []

        for gpu_id, tasks in gpu_tasks.items():
            if not tasks:
                continue

            # Create temporary task file
            temp_file = self.output_dir / f"temp_tasks_gpu_{gpu_id}.json"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(tasks, f, ensure_ascii=False)
            temp_files.append(temp_file)

            # Start GPU processes
            process = self._start_gpu_process(gpu_id, model_type, model_variant, config_path, temp_file, seed)
            processes.append(process)

        # Monitor process completion
        self._monitor_processes(processes, temp_files)

        # Merge results
        self._merge_results()

        print("Distributed generation task completed!")

    def _filter_completed_tasks(self, prompt_data: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
        """
        Filter completed tasks based on actual image file existence

        Args:
            prompt_data: All prompt data

        Returns:
            List of incomplete prompt data
        """
        images_dir = self.output_dir / "images"
        remaining_tasks = []
        completed_count = 0

        for i, (prompt, image_id) in enumerate(prompt_data):
            # Generate expected filename (consistent with logic in generate.py)
            if image_id:
                filename = f"{image_id}.png"
            else:
                safe_filename = sanitize_filename(prompt)
                filename = f"{safe_filename}_{i:04d}.png"

            # Check if image file already exists
            image_path = images_dir / filename
            if image_path.exists() and image_path.is_file():
                completed_count += 1
                print(f"Skip completed task: {filename}")
            else:
                remaining_tasks.append((prompt, image_id))

        print(f"Found {completed_count}  completed tasks, {len(remaining_tasks)}  pending tasks")
        return remaining_tasks

    def _allocate_tasks_to_gpus(
        self, tasks: List[Tuple[str, Optional[str]]], gpu_ids: List[int]
    ) -> Dict[int, List[Tuple[str, Optional[str]]]]:
        """Allocate tasks to GPUs"""
        num_gpus = len(gpu_ids)
        tasks_per_gpu = math.ceil(len(tasks) / num_gpus)

        allocation = {}
        for i, gpu_id in enumerate(gpu_ids):
            start_idx = i * tasks_per_gpu
            end_idx = min((i + 1) * tasks_per_gpu, len(tasks))
            allocation[gpu_id] = tasks[start_idx:end_idx]

        return allocation

    def _start_gpu_process(
        self, gpu_id: int, model_type: str, model_variant: str, config_path: str, task_file: Path, seed: int
    ) -> subprocess.Popen:
        """Start single GPU process"""

        # Use path manager to get absolute path
        absolute_config_path = paths.get_config_path(config_path)

        # Build command
        cmd = [
            "python",
            "-m",
            "evaluation.generate",  # Use module invocation method
            "--model",
            f"{model_type}/{model_variant}",
            "--config",
            str(absolute_config_path),
            "--prompt_data_file",
            str(task_file.absolute()),
            "--output_dir",
            str(self.output_dir.absolute()),
            "--temp_json_filename",
            f"temp_mappings_gpu_{gpu_id}.json",
            "--seed",
            str(seed),
        ]

        print(f"GPU {gpu_id} Start command: {' '.join(cmd)}")
        print(f"GPU {gpu_id} Using config file: {absolute_config_path}")

        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Start process
        log_file = self.output_dir / f"gpu_{gpu_id}.log"
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(paths.project_root),  # Set working directory to project root
            )

        print(f"GPU {gpu_id} Task started, log: {log_file}")
        return process

    def _monitor_processes(self, processes: List[subprocess.Popen], temp_files: List[Path]) -> None:
        """Monitor process completion"""
        print("\nWaiting for all GPU tasks to complete...")

        for process in processes:
            process.wait()

        print("All GPU tasks completed")

        # Clean up temporary files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                print(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                print(f"Failed to delete temporary file {temp_file}: {e}")

    def _merge_results(self) -> None:
        """Merge results from all GPUs"""
        all_mappings = []

        # First read existing mapping file (key modification for checkpoint resumption)
        final_mapping_file = self.output_dir / "prompts.json"
        if final_mapping_file.exists():
            try:
                with open(final_mapping_file, "r", encoding="utf-8") as f:
                    existing_mappings = json.load(f)
                print(f"Read existing {len(existing_mappings)} mapping records")
                all_mappings.extend(existing_mappings)
            except Exception as e:
                print(f"Failed to read existing mapping file: {e}")

        # Process newly generated mapping files
        mapping_files = list(self.output_dir.glob("temp_mappings_gpu_*.json"))
        new_mappings_count = 0

        for mapping_file in mapping_files:
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    mappings = json.load(f)
                all_mappings.extend(mappings)
                new_mappings_count += len(mappings)

                # Delete temporary file
                mapping_file.unlink()
                print(f"Processed and deleted: {mapping_file}")

            except Exception as e:
                print(f"Failed to process mapping file {mapping_file}: {e}")

        # Deduplication: deduplicate based on image path, keep latest mapping
        unique_mappings = {}
        for mapping in all_mappings:
            image_path = mapping.get("image_path", mapping.get("image", ""))
            unique_mappings[image_path] = mapping

        final_mappings = list(unique_mappings.values())

        # Save final mapping file
        with open(final_mapping_file, "w", encoding="utf-8") as f:
            json.dump(final_mappings, f, indent=2, ensure_ascii=False)

        print(
            f"Merge completed: added {new_mappings_count}  mappings, total {len(final_mappings)}  mappings to {final_mapping_file}"
        )

        # Create simple completion status file
        completion_info = {
            "completed_images": len(final_mappings),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images_dir": "images",
        }

        completion_file = self.output_dir / "completion_status.json"
        with open(completion_file, "w", encoding="utf-8") as f:
            json.dump(completion_info, f, indent=2, ensure_ascii=False)

    def get_status(self) -> Dict[str, Any]:
        """Get task status"""
        images_dir = self.output_dir / "images"

        # Count generated image files
        completed_images = 0
        if images_dir.exists():
            completed_images = len([f for f in images_dir.glob("*.png") if f.is_file()])

        # Read completion status file (if exists)
        completion_file = self.output_dir / "completion_status.json"
        last_update = None
        if completion_file.exists():
            try:
                last_update = time.ctime(completion_file.stat().st_mtime)
            except:
                pass

        return {
            "output_dir": str(self.output_dir),
            "completed_images": completed_images,
            "images_dir_exists": images_dir.exists(),
            "last_update": last_update,
            "images_directory": str(images_dir),
        }
