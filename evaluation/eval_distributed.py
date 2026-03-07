#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import threading
import time
from typing import List
import sys


def get_available_gpus() -> List[int]:
    """
    Get list of available GPUs.

    Returns:
        List[int]: List of available GPU IDs
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_ids = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
        return gpu_ids
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Cannot detect GPU, will use CPU")
        return []


def run_evaluation_task(gpu_id: int, paths: List[str], metrics: str, additional_args: List[str]) -> None:
    """
    Run evaluation task on specified GPU.

    Args:
        gpu_id: GPU ID
        paths: List of paths to evaluate
        metrics: Metric names
        additional_args: Additional command line arguments
    """
    for path in paths:
        print(f"GPU {gpu_id}: Starting evaluation {path}")

        # Ensure evaluation directory exists
        eval_dir = os.path.join(path, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # Build command
        cmd = [sys.executable, "-m", "evaluation.eval", "--generated_path", path, "--metrics", metrics]

        # Add additional arguments
        cmd.extend(additional_args)

        # Set environment variables
        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set output file
        log_file = os.path.join(eval_dir, "eval_output.log")

        try:
            # Run command and log output
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    cwd=os.getcwd(),
                )

                # Write output in real-time
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    # Also print to console
                    print(f"GPU {gpu_id}: {line.rstrip()}")

                process.wait()

                f.write(f"\n\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Return code: {process.returncode}\n")

            if process.returncode == 0:
                print(f"GPU {gpu_id}: Successfully completed {path}")
            else:
                print(f"GPU {gpu_id}: Evaluation failed {path} (return code: {process.returncode})")

        except Exception as e:
            error_msg = f"GPU {gpu_id}: Error occurred while running {path}: {e}"
            print(error_msg)

            # Log error to log file
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\nError: {error_msg}\n")
            except:
                pass


def distribute_paths(paths: List[str], num_workers: int) -> List[List[str]]:
    """
    Distribute path list to worker processes.

    Args:
        paths: List of paths
        num_workers: Number of worker processes

    Returns:
        List[List[str]]: List of paths assigned to each worker process
    """
    if num_workers <= 0:
        return [paths]

    # Calculate number of paths each worker should handle
    paths_per_worker = len(paths) // num_workers
    remainder = len(paths) % num_workers

    distributed_paths = []
    start_idx = 0

    for i in range(num_workers):
        # First remainder workers get one extra path
        end_idx = start_idx + paths_per_worker + (1 if i < remainder else 0)
        distributed_paths.append(paths[start_idx:end_idx])
        start_idx = end_idx

    return distributed_paths


def main():
    parser = argparse.ArgumentParser(description="Distributed image generation evaluation script")

    parser.add_argument(
        "--generated_paths", type=str, nargs="+", required=True, help="List of generated image directory paths"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Metrics to calculate, comma-separated, e.g. AestheticScore,CLIPScore",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="Number of GPUs to use, defaults to all available GPUs"
    )
    parser.add_argument(
        "--device_ids", type=str, default=None, help="Specify GPU IDs to use, comma-separated, e.g. 0,1,2,3"
    )
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")

    args = parser.parse_args()

    # Check if all paths exist
    missing_paths = []
    for path in args.generated_paths:
        if not os.path.exists(path):
            missing_paths.append(path)

    if missing_paths:
        print(f"The following paths do not exist:")
        for path in missing_paths:
            print(f"  - {path}")
        print(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"{len(missing_paths)} paths do not exist, please check if paths are correct")

    # Note: Real image paths are now automatically read from config files by each evaluation task
    print("Evaluation tasks will automatically read real image paths from config files")

    # Determine GPUs to use
    if args.device_ids:
        # Parse comma-separated GPU ID string
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.device_ids.split(",")]
        print(f"Using specified GPUs: {gpu_ids}")
    else:
        available_gpus = get_available_gpus()
        if not available_gpus:
            gpu_ids = [-1]  # Use CPU
            print("No GPU detected, will use CPU")
        else:
            if args.num_gpus:
                gpu_ids = available_gpus[: args.num_gpus]
            else:
                gpu_ids = available_gpus
            print(f"Detected available GPUs: {available_gpus}")
            print(f"Will use GPUs: {gpu_ids}")

    # Distribute paths to each GPU
    distributed_paths = distribute_paths(args.generated_paths, len(gpu_ids))

    print(f"\nTask distribution:")
    for i, (gpu_id, paths) in enumerate(zip(gpu_ids, distributed_paths)):
        device_name = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
        print(f"  {device_name}: {len(paths)} paths")
        for path in paths:
            print(f"    - {path}")

    # Prepare additional command line arguments
    additional_args = []
    if args.batch_size != 250:
        additional_args.extend(["--batch_size", str(args.batch_size)])

    # Start threads
    threads = []
    start_time = time.time()

    print(f"\nStarting distributed evaluation...")

    for gpu_id, paths in zip(gpu_ids, distributed_paths):
        if paths:  # Only start threads for GPUs with tasks
            thread = threading.Thread(target=run_evaluation_task, args=(gpu_id, paths, args.metrics, additional_args))
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time
    print(f"\nAll evaluation tasks completed! Total time: {total_time:.2f} seconds")
    print(f"Please check evaluation/eval_output.log files in each path for detailed output")


if __name__ == "__main__":
    main()
