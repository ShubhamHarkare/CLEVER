#!/usr/bin/env python3
"""
Latency Sensitivity Analysis.

This script loads the routing evaluation results and recalculates the 
latency savings under different LLM latency assumptions (T_llm) to 
demonstrate the sensitivity of the break-even points, answering
reviewer feedback on static T_llm assumptions.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.router.cost_model import CostModel


def main():
    parser = argparse.ArgumentParser(description="Latency Sensitivity Analysis")
    parser.add_argument(
        "--input", 
        type=str, 
        default="results/routing/routing_eval_multi_seed.json",
        help="Path to routing evaluation JSON"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run routing evaluation first.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    # Determine if it's multi-seed or single-seed and extract random strategy baseline
    if "aggregated" in data:
        strat_data = data["aggregated"].get("random", {})
        if not strat_data:
            print("Could not find 'random' strategy in aggregated results.")
            return
        hit_rate = strat_data.get("test_hit_rate", {}).get("mean", 0.0)
    else:
        strat_data = data.get("random", {})
        if not strat_data:
            print("Could not find 'random' strategy in results.")
            return
        hit_rate = strat_data.get("adaptive", {}).get("test_hit_rate", 0.0)
        
    print("=" * 60)
    print(f"LATENCY SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Base Cache Hit Rate: {hit_rate:.2%}")
    print()
    print("Assumptions:")
    print("  - Cache latency (T_cache): 0.5 ms (approx)")
    print()
    print(f"{'LLM Latency (ms)':<20} | {'Effective Letency (ms)':<25} | {'Savings (%)':<15}")
    print("-" * 65)

    # Different LLM latency assumptions
    llm_latencies = [200, 500, 1000, 2000, 5000]

    for t_llm in llm_latencies:
        model = CostModel(
            llm_latency_ms=t_llm, 
            cache_latency_ms=0.5, # Assuming nominal ~0.5ms HNSW lookup
        )
        savings = model.cost_savings(hit_rate=hit_rate)
        
        eff_lat = savings["avg_latency_ms"]
        sav_pct = savings["latency_saving_pct"]
        
        print(f"{t_llm:<20} | {eff_lat:<25.2f} | {sav_pct:>6.1f} %")
        
    print("-" * 65)

if __name__ == "__main__":
    main()
