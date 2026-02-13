# main.py
"""
MARS Framework - Unified Challenge Runner
Reads challenge context from text files only.
NO model suggestions - agent discovers everything.
"""
from pathlib import Path
import argparse
import json

from core.config import Config
from core.challenge_loader import ChallengeLoader
from mle.eda_agent import EDAAgent
from orchestrator import MARSOrchestrator

def main():
    parser = argparse.ArgumentParser(description="MARS - Unified Challenge Runner")
    parser.add_argument(
        "--challenge", 
        type=str, 
        required=True,
        help="Challenge name (e.g., 'otto_group', 'titanic', 'spaceship')"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Path to challenge data directory"
    )
    parser.add_argument(
        "--time-budget", 
        type=int, 
        default=3600,
        help="Time budget in seconds (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (default: ./working)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Config.WORKING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MARS Framework - Autonomous ML System")
    print("="*70)
    print(f"Challenge: {args.challenge}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Budget: {args.time_budget}s ({args.time_budget/3600:.1f}h)")
    print("="*70 + "\n")
    
    # ========================================
    # STEP 1: Load Challenge (Pure Context)
    # ========================================
    print("="*70)
    print("STEP 1: Loading Challenge Context")
    print("="*70)
    
    try:
        loader = ChallengeLoader(args.challenge, data_dir)
        
        # Get ONLY the problem description (no model hints)
        problem_description = loader.get_problem_description()
        
        print(f"✓ Loaded challenge: {args.challenge}")
        print(f"✓ Description: {len(problem_description)} characters")
        print(f"✓ Metric: {loader.metric_name} ({loader.metric_direction})")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"\nAvailable challenges:")
        challenges_dir = Path("challenges")
        if challenges_dir.exists():
            for file in challenges_dir.glob("*.txt"):
                print(f"  - {file.stem}")
        return
    
    # ========================================
    # STEP 2: Load and Prepare Data
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: Data Preparation")
    print("="*70)
    
    # Load data
    data = loader.load_data()
    
    if 'train' not in data:
        print("✗ Error: Training data not found")
        return
    
    # Prepare splits
    metadata_dir = output_dir / "metadata"
    splits = loader.prepare_splits(
        train_df=data['train'],
        metadata_dir=metadata_dir,
        validation_ratio=0.2
    )
    
    # Copy test data if exists
    if 'test' in data:
        test_path = metadata_dir / "test.csv"
        data['test'].to_csv(test_path, index=False)
        splits['test'] = test_path
        print(f"  ✓ Test: {len(data['test'])} samples")
    
    # ========================================
    # STEP 3: Quick EDA
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: Exploratory Data Analysis")
    print("="*70)
    
    eda_agent = EDAAgent()
    eda_report = eda_agent.analyze_data(
        train_csv=splits["train"],
        target_column=loader.target_column or "target",
        problem_description=problem_description
    )
    
    # Save EDA
    eda_path = metadata_dir / "eda_report.md"
    with open(eda_path, 'w') as f:
        f.write(eda_report)
    print(f"  ✓ EDA saved: {eda_path}")
    
    # ========================================
    # STEP 4: Run MARS (Pure Discovery)
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: MARS Autonomous Search")
    print("="*70)
    print("Agent will discover models and approaches independently.")
    print(f"Time budget: {args.time_budget}s")
    print("="*70 + "\n")
    
    # Get metric direction
    _, lower_is_better = loader.get_metric_info()

    # Initialize orchestrator with ONLY problem description
    # NO model suggestions - agent figures everything out
    orchestrator = MARSOrchestrator(
        problem_description=problem_description,  # ← ONLY CONTEXT
        eda_report=eda_report,
        metadata_dir=metadata_dir,
        data_dir=data_dir,
        time_budget=args.time_budget,
        working_dir=output_dir,
        lower_is_better=lower_is_better,
    )
    
    # Run search
    best_node = orchestrator.run()
    
    # ========================================
    # STEP 5: Save Results
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: Results")
    print("="*70)
    
    if best_node and best_node.solution:
        # Save best solution code
        best_dir = output_dir / "best_solution"
        best_dir.mkdir(exist_ok=True)
        
        for filename, code in best_node.solution.get_all_files().items():
            filepath = best_dir / filename
            with open(filepath, 'w') as f:
                f.write(code)
            print(f"  ✓ Saved: {filename}")
        
        # Save solution info
        metric_name, lower_is_better = loader.get_metric_info()
        
        info = {
            "challenge": args.challenge,
            "node_id": best_node.id,
            "metric_name": metric_name,
            "metric_value": best_node.metric_value,
            "lower_is_better": lower_is_better,
            "execution_time_seconds": best_node.execution_time,
            "idea": best_node.solution.idea,
            "timestamp": str(best_node.id),
        }
        
        info_path = best_dir / "solution_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n{'='*70}")
        print("BEST SOLUTION FOUND")
        print(f"{'='*70}")
        print(f"Metric ({metric_name}): {best_node.metric_value:.6f}")
        print(f"Direction: {'Lower is better' if lower_is_better else 'Higher is better'}")
        print(f"Execution time: {best_node.execution_time:.1f}s")
        print(f"Solution saved to: {best_dir}")
        print(f"{'='*70}")
        
    else:
        print("\nNo valid solution found within time budget")
        print("   Suggestions:")
        print("   - Increase time budget (--time-budget 7200)")
        print("   - Check logs in working directory")
        print("   - Verify data format matches challenge requirements")
    
    # ========================================
    # STEP 6: Final Statistics
    # ========================================
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    
    stats = orchestrator.mcts.get_statistics()
    print(f"Total iterations: {stats['iterations']}")
    print(f"Nodes explored: {stats['nodes_explored']}")
    print(f"Valid solutions: {stats['valid_nodes']}")
    print(f"Buggy attempts: {stats['buggy_nodes']}")
    
    # Lesson statistics
    lesson_stats = orchestrator.lesson_pool.get_statistics()
    print(f"\nLessons learned:")
    print(f"  Solution lessons: {lesson_stats['solution_lessons']}")
    print(f"  Debug lessons: {lesson_stats['debug_lessons']}")
    
    print("\n" + "="*70)
    print("MARS RUN COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()