#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

# Try to import pandas, fallback to basic CSV if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic CSV output")


class SD3ComparisonAnalyzer:
    """SD3 Methods Comparison Analysis"""
    
    def __init__(self, output_dir: str, save_dir: str = None):
        """
        Initialize analyzer
        
        Args:
            output_dir: Output directory containing method results
            save_dir: Directory to save analysis results (default: same as output_dir)
        """
        self.output_dir = Path(output_dir)
        self.save_dir = Path(save_dir) if save_dir else self.output_dir
        
        # Expected methods and metrics (updated to match actual directory names)
        self.methods = ["cfg_flux_cfg", "cads_flux_cads", "icg_flux_icg", "cdg_flux_cdg"]
        self.method_labels = {
            "cfg_flux_cfg": "CFG",
            "cads_flux_cads": "CADS", 
            "icg_flux_icg": "ICG",
            "cdg_flux_cdg": "CDG"
        }
        self.metrics = ["FID", "CLIPScore", "AestheticScore", "VQAScore"]
        
        # Data storage
        self.data = {}
    
    def load_results(self) -> None:
        """Load evaluation results from all method directories"""
        print("Loading evaluation results...")
        
        loaded_count = 0
        failed_count = 0
        
        for method in self.methods:
            method_dir = self.output_dir / method
            result_file = method_dir / "evaluation" / "metrics_results.json"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    # Extract metric scores
                    method_data = {}
                    for metric in self.metrics:
                        if metric in results.get("results", {}):
                            value = results["results"][metric]
                            # Handle different result formats
                            if isinstance(value, dict) and 'mean' in value:
                                method_data[metric] = value['mean']
                            else:
                                method_data[metric] = value
                        else:
                            print(f"  Warning: Metric {metric} not found in {method}")
                            method_data[metric] = None
                    
                    self.data[method] = method_data
                    loaded_count += 1
                    print(f"  ✓ Loaded: {self.method_labels[method]} ({method})")
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {method}: {e}")
                    failed_count += 1
            else:
                print(f"  ✗ Result file not found: {result_file}")
                failed_count += 1
        
        print(f"\nLoading summary:")
        print(f"  Successfully loaded: {loaded_count}/{len(self.methods)}")
        print(f"  Failed: {failed_count}")
        
        if loaded_count == 0:
            raise ValueError("No content evaluation results found")
    
    def create_comparison_table(self) -> List[Dict[str, str]]:
        """Create comparison table with methods as rows and metrics as columns"""
        print("\nCreating comparison table...")
        
        # Prepare table data
        table_data = []
        
        for method in self.methods:
            if method in self.data:
                row = {"Method": self.method_labels[method]}
                for metric in self.metrics:
                    value = self.data[method].get(metric)
                    if value is not None:
                        # Format numbers to 4 decimal places
                        row[metric] = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    else:
                        row[metric] = "N/A"
                table_data.append(row)
            else:
                # Add row with N/A values if method data not found
                row = {"Method": self.method_labels[method]}
                for metric in self.metrics:
                    row[metric] = "N/A"
                table_data.append(row)
        
        return table_data
    
    def save_results(self, table_data: List[Dict[str, str]]) -> None:
        """Save results to CSV file"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV file
        csv_file = self.save_dir / "sd3_methods_comparison.csv"
        
        if HAS_PANDAS:
            # Use pandas if available
            df = pd.DataFrame(table_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✓ Comparison table saved: {csv_file}")
            
            # Print table to console
            print("\nSD3 Methods Comparison Results:")
            print("=" * 60)
            print(df.to_string(index=False))
            print("=" * 60)
        else:
            # Use basic CSV writer
            if table_data:
                fieldnames = ["Method"] + self.metrics
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(table_data)
                print(f"✓ Comparison table saved: {csv_file}")
                
                # Print table to console
                self._print_table_basic(table_data)
    
    def _print_table_basic(self, table_data: List[Dict[str, str]]) -> None:
        """Print table using basic formatting"""
        print("\nSD3 Methods Comparison Results:")
        print("=" * 60)
        
        # Print header
        header = ["Method"] + self.metrics
        print(" | ".join(f"{col:>12}" for col in header))
        print("-" * 60)
        
        # Print data rows
        for row in table_data:
            values = [row.get(col, "N/A") for col in header]
            print(" | ".join(f"{val:>12}" for val in values))
        
        print("=" * 60)
    
    def run_analysis(self) -> None:
        """Run complete analysis"""
        print("=== SD3 Methods Comparison Analysis ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Save directory: {self.save_dir}")
        
        # Load results
        self.load_results()
        
        # Create comparison table
        table_data = self.create_comparison_table()
        
        # Save results
        self.save_results(table_data)
        
        print("\n✓ Analysis completed successfully!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SD3 Methods Comparison Analysis')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory containing method results')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save analysis results (default: same as output_dir)')
    
    args = parser.parse_args()
    
    try:
        analyzer = SD3ComparisonAnalyzer(args.output_dir, args.save_dir)
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
