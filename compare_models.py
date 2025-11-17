"""
Model Comparison Script
Compares all trained models and identifies the best performing one
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_experiment_results(member_name="Annemarie"):
    """
    Load all experiment results from JSON files
    """
    results_dir = f"results/{member_name}_logs"
    
    # Find the most recent hyperparameter results file
    json_files = list(Path(results_dir).glob("hyperparameter_results_*.json"))
    
    if not json_files:
        print(f"No hyperparameter results found in {results_dir}")
        return None
    
    # Use the most recent file (or first one found)
    results_file = json_files[0]
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def load_training_logs(member_name="Annemarie"):
    """
    Load training logs for all experiments to get detailed training history
    """
    logs_dir = f"logs/{member_name}_logs"
    training_logs = {}
    
    log_files = list(Path(logs_dir).glob("training_logs_*.json"))
    
    for log_file in log_files:
        experiment_name = log_file.stem.replace("training_logs_", "")
        try:
            with open(log_file, 'r') as f:
                training_logs[experiment_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {log_file}: {e}")
    
    return training_logs


def compare_models(results, member_name="Annemarie"):
    """
    Compare all models and identify the best performing one
    """
    if not results:
        print("No results to compare!")
        return None
    
    # Sort by mean reward (descending)
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    
    print("\n" + "="*120)
    print("MODEL COMPARISON RESULTS")
    print("="*120)
    print(f"\n{'Rank':<6} {'Experiment':<30} {'Mean Reward':<15} {'Std Reward':<15} {'Learning Rate':<15} {'Gamma':<10} {'Batch Size':<12} {'Behavior':<40}")
    print("-"*120)
    
    for rank, result in enumerate(sorted_results, 1):
        if 'error' not in result:
            print(f"{rank:<6} "
                  f"{result['experiment']:<30} "
                  f"{result['mean_reward']:<15.2f} "
                  f"{result['std_reward']:<15.2f} "
                  f"{result['lr']:<15.2e} "
                  f"{result['gamma']:<10.3f} "
                  f"{result['batch_size']:<12} "
                  f"{result['behavior']:<40}")
        else:
            print(f"{rank:<6} {result['experiment']:<30} ERROR: {result['error']}")
    
    print("="*120)
    
    # Identify best model
    best_model = sorted_results[0]
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model['experiment']}")
    print(f"   Mean Reward: {best_model['mean_reward']:.2f} ± {best_model['std_reward']:.2f}")
    print(f"   Hyperparameters:")
    print(f"     - Learning Rate: {best_model['lr']:.2e}")
    print(f"     - Gamma: {best_model['gamma']:.3f}")
    print(f"     - Batch Size: {best_model['batch_size']}")
    print(f"     - Epsilon Start: {best_model['eps_start']:.2f}")
    print(f"     - Epsilon End: {best_model['eps_end']:.3f}")
    print(f"     - Exploration Fraction: {best_model['exploration_fraction']:.2f}")
    print(f"   Behavior: {best_model['behavior']}")
    
    return sorted_results, best_model


def create_comparison_visualizations(results, member_name="Annemarie"):
    """
    Create visualizations comparing all models
    """
    if not results:
        return
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No valid results to visualize!")
        return
    
    # Sort by mean reward
    sorted_results = sorted(valid_results, key=lambda x: x['mean_reward'], reverse=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Mean Reward Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    experiments = [r['experiment'].replace('Experiment_', 'Exp_') for r in sorted_results]
    rewards = [r['mean_reward'] for r in sorted_results]
    std_rewards = [r['std_reward'] for r in sorted_results]
    
    bars = ax1.barh(range(len(experiments)), rewards, xerr=std_rewards, capsize=5)
    ax1.set_yticks(range(len(experiments)))
    ax1.set_yticklabels(experiments, fontsize=8)
    ax1.set_xlabel('Mean Reward', fontsize=10)
    ax1.set_title('Mean Reward Comparison (with Std Dev)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Color the best model
    bars[0].set_color('gold')
    
    # 2. Learning Rate vs Reward
    ax2 = plt.subplot(2, 3, 2)
    lrs = [r['lr'] for r in sorted_results]
    ax2.scatter(lrs, rewards, s=100, alpha=0.6, c=range(len(rewards)), cmap='viridis')
    ax2.set_xlabel('Learning Rate', fontsize=10)
    ax2.set_ylabel('Mean Reward', fontsize=10)
    ax2.set_title('Learning Rate vs Reward', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Gamma vs Reward
    ax3 = plt.subplot(2, 3, 3)
    gammas = [r['gamma'] for r in sorted_results]
    ax3.scatter(gammas, rewards, s=100, alpha=0.6, c=range(len(rewards)), cmap='viridis')
    ax3.set_xlabel('Gamma (Discount Factor)', fontsize=10)
    ax3.set_ylabel('Mean Reward', fontsize=10)
    ax3.set_title('Gamma vs Reward', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Batch Size vs Reward
    ax4 = plt.subplot(2, 3, 4)
    batch_sizes = [r['batch_size'] for r in sorted_results]
    ax4.scatter(batch_sizes, rewards, s=100, alpha=0.6, c=range(len(rewards)), cmap='viridis')
    ax4.set_xlabel('Batch Size', fontsize=10)
    ax4.set_ylabel('Mean Reward', fontsize=10)
    ax4.set_title('Batch Size vs Reward', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Exploration Fraction vs Reward
    ax5 = plt.subplot(2, 3, 5)
    expl_fractions = [r['exploration_fraction'] for r in sorted_results]
    ax5.scatter(expl_fractions, rewards, s=100, alpha=0.6, c=range(len(rewards)), cmap='viridis')
    ax5.set_xlabel('Exploration Fraction', fontsize=10)
    ax5.set_ylabel('Mean Reward', fontsize=10)
    ax5.set_title('Exploration Fraction vs Reward', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Hyperparameter Heatmap (Top 5 models)
    ax6 = plt.subplot(2, 3, 6)
    top_5 = sorted_results[:5]
    top_5_names = [r['experiment'].replace('Experiment_', 'Exp_') for r in top_5]
    
    # Normalize hyperparameters for visualization
    hyperparams = np.array([
        [r['lr'] * 10000, r['gamma'] * 100, r['batch_size'] / 10, 
         r['exploration_fraction'] * 100, r['mean_reward']] 
        for r in top_5
    ])
    
    im = ax6.imshow(hyperparams, aspect='auto', cmap='YlOrRd')
    ax6.set_xticks(range(5))
    ax6.set_xticklabels(['LR×10k', 'γ×100', 'Batch/10', 'Expl%', 'Reward'], fontsize=8)
    ax6.set_yticks(range(len(top_5_names)))
    ax6.set_yticklabels(top_5_names, fontsize=8)
    ax6.set_title('Top 5 Models - Normalized Hyperparameters', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = f"results/{member_name}_logs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison visualization saved to: {output_path}")
    plt.close()


def create_training_curves_comparison(training_logs, member_name="Annemarie"):
    """
    Compare training curves across all experiments
    """
    if not training_logs:
        print("No training logs available for comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot rewards over time
    for exp_name, log_data in training_logs.items():
        if 'rewards' in log_data and len(log_data['rewards']) > 0:
            rewards = log_data['rewards']
            ax1.plot(rewards, label=exp_name.replace('Experiment_', 'Exp_'), alpha=0.7, linewidth=2)
    
    ax1.set_xlabel('Checkpoint', fontsize=11)
    ax1.set_ylabel('Mean Episode Reward', fontsize=11)
    ax1.set_title('Training Rewards Over Time - All Experiments', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths over time
    for exp_name, log_data in training_logs.items():
        if 'episode_lengths' in log_data and len(log_data['episode_lengths']) > 0:
            lengths = log_data['episode_lengths']
            ax2.plot(lengths, label=exp_name.replace('Experiment_', 'Exp_'), alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Checkpoint', fontsize=11)
    ax2.set_ylabel('Mean Episode Length', fontsize=11)
    ax2.set_title('Episode Length Over Time - All Experiments', fontsize=13, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = f"results/{member_name}_logs"
    output_path = f"{output_dir}/training_curves_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves comparison saved to: {output_path}")
    plt.close()


def save_comparison_report(sorted_results, best_model, member_name="Annemarie"):
    """
    Save a detailed comparison report to a text file
    """
    output_dir = f"results/{member_name}_logs"
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/model_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*120 + "\n\n")
        f.write(f"Member: {member_name}\n")
        f.write(f"Total Experiments: {len(sorted_results)}\n")
        f.write(f"Date: {os.popen('date /t').read().strip() if os.name == 'nt' else os.popen('date').read().strip()}\n\n")
        
        f.write("="*120 + "\n")
        f.write("🏆 BEST PERFORMING MODEL\n")
        f.write("="*120 + "\n\n")
        f.write(f"Experiment: {best_model['experiment']}\n")
        f.write(f"Mean Reward: {best_model['mean_reward']:.2f} ± {best_model['std_reward']:.2f}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  - Learning Rate: {best_model['lr']:.2e}\n")
        f.write(f"  - Gamma: {best_model['gamma']:.3f}\n")
        f.write(f"  - Batch Size: {best_model['batch_size']}\n")
        f.write(f"  - Epsilon Start: {best_model['eps_start']:.2f}\n")
        f.write(f"  - Epsilon End: {best_model['eps_end']:.3f}\n")
        f.write(f"  - Exploration Fraction: {best_model['exploration_fraction']:.2f}\n")
        f.write(f"  - Epsilon Decay Per Step: {best_model['epsilon_decay_per_step']:.2e}\n")
        f.write(f"  - Behavior: {best_model['behavior']}\n\n")
        
        f.write("="*120 + "\n")
        f.write("RANKED RESULTS (Best to Worst)\n")
        f.write("="*120 + "\n\n")
        f.write(f"{'Rank':<6} {'Experiment':<30} {'Mean Reward':<15} {'Std Reward':<15} {'LR':<12} {'Gamma':<8} {'Batch':<8} {'Behavior':<40}\n")
        f.write("-"*120 + "\n")
        
        for rank, result in enumerate(sorted_results, 1):
            if 'error' not in result:
                f.write(f"{rank:<6} "
                       f"{result['experiment']:<30} "
                       f"{result['mean_reward']:<15.2f} "
                       f"{result['std_reward']:<15.2f} "
                       f"{result['lr']:<12.2e} "
                       f"{result['gamma']:<8.3f} "
                       f"{result['batch_size']:<8} "
                       f"{result['behavior']:<40}\n")
            else:
                f.write(f"{rank:<6} {result['experiment']:<30} ERROR: {result['error']}\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("="*120 + "\n\n")
        
        valid_results = [r for r in sorted_results if 'error' not in r]
        if valid_results:
            rewards = [r['mean_reward'] for r in valid_results]
            f.write(f"Highest Reward: {max(rewards):.2f}\n")
            f.write(f"Lowest Reward: {min(rewards):.2f}\n")
            f.write(f"Average Reward: {np.mean(rewards):.2f}\n")
            f.write(f"Median Reward: {np.median(rewards):.2f}\n")
            f.write(f"Standard Deviation: {np.std(rewards):.2f}\n")
            f.write(f"Range: {max(rewards) - min(rewards):.2f}\n")
    
    print(f"📄 Comparison report saved to: {report_path}")


def main():
    """
    Main function to run model comparison
    """
    print("\n" + "="*120)
    print("MODEL COMPARISON TOOL")
    print("="*120)
    
    # You can change this to compare different members' experiments
    member_name = "Annemarie"
    
    # Load results
    print(f"\nLoading experiment results for {member_name}...")
    results = load_experiment_results(member_name)
    
    if not results:
        print("No results found!")
        return
    
    # Load training logs
    print("Loading training logs...")
    training_logs = load_training_logs(member_name)
    
    # Compare models
    print("\nComparing models...")
    sorted_results, best_model = compare_models(results, member_name)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_visualizations(results, member_name)
    
    if training_logs:
        create_training_curves_comparison(training_logs, member_name)
    
    # Save report
    print("\nGenerating comparison report...")
    save_comparison_report(sorted_results, best_model, member_name)
    
    print("\n" + "="*120)
    print("COMPARISON COMPLETE!")
    print("="*120)
    print(f"\nCheck the following files in results/{member_name}_logs/:")
    print("  - model_comparison.png (visual comparison)")
    print("  - training_curves_comparison.png (training history)")
    print("  - model_comparison_report.txt (detailed report)")


if __name__ == "__main__":
    main()

