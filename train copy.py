"""
DQN Training Script for Atari Breakout Environment
Trains a DQN agent using Stable Baselines3 with hyperparameter tuning
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
import os
from datetime import datetime

from environment import setup_environment

# Create directories for saving models and logs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)


class TrainingCallback(BaseCallback):
    """
    Custom callback for logging training metrics
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log episode rewards
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                self.rewards.append(mean_reward)
                self.episode_lengths.append(mean_length)
                
                if self.verbose > 0:
                    print(f"Step: {self.n_calls}, Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}")
        
        return True
    
    def save_logs(self, filename):
        """Save training logs to file"""
        logs = {
            'rewards': self.rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(os.path.join(self.log_dir, filename), 'w') as f:
            json.dump(logs, f)


def train_dqn_agent(
    policy_type="CnnPolicy",
    total_timesteps=1000000,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    buffer_size=100000,
    learning_starts=50000,
    target_update_interval=1000,
    train_freq=4,
    experiment_name="default",
    save_model=True
):
    """
    Train a DQN agent with specified hyperparameters
    
    Args:
        policy_type: "CnnPolicy" or "MlpPolicy"
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        batch_size: Batch size for training
        exploration_fraction: Fraction of training for epsilon decay
        exploration_initial_eps: Initial epsilon value
        exploration_final_eps: Final epsilon value
        buffer_size: Size of replay buffer
        learning_starts: Steps before training starts
        target_update_interval: Steps between target network updates
        train_freq: Frequency of training updates
        experiment_name: Name for saving results
        save_model: Whether to save the trained model
    
    Returns:
        model: Trained DQN model
        callback: Training callback with logged metrics
    """
    
    print(f"\n{'='*80}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Policy: {policy_type}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epsilon Start: {exploration_initial_eps}")
    print(f"  Epsilon End: {exploration_final_eps}")
    print(f"  Exploration Fraction: {exploration_fraction}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = setup_environment()
    
    # Create DQN model
    model = DQN(
        policy_type,
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=f"./logs/{experiment_name}"
    )
    
    # Create callback
    callback = TrainingCallback(
        check_freq=10000,
        log_dir="logs",
        verbose=1
    )
    
    # Train the model
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10
    )
    
    # Save the model
    if save_model:
        model_path = f"models/dqn_model_{experiment_name}.zip"
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save logs
        callback.save_logs(f"training_logs_{experiment_name}.json")
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot training curves
    plot_training_curves(callback, experiment_name)
    
    # --- Clean up to prevent memory leaks ---
    env.close()
    del model
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return None, callback, mean_reward, std_reward

def plot_training_curves(callback, experiment_name):
    """
    Plot and save training curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    if len(callback.rewards) > 0:
        ax1.plot(callback.rewards)
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Training Rewards Over Time')
        ax1.grid(True)
    
    # Plot episode lengths
    if len(callback.episode_lengths) > 0:
        ax2.plot(callback.episode_lengths)
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Mean Episode Length')
        ax2.set_title('Episode Length Over Time')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/training_curves_{experiment_name}.png')
    plt.close()
    print(f"Training curves saved to: results/training_curves_{experiment_name}.png")


def compare_policies():
    """
    Compare MLP and CNN policies
    Note: For Atari games, CNN is strongly recommended due to visual input
    """
    print("\n" + "="*80)
    print("COMPARING POLICIES: MLP vs CNN")
    print("="*80)
    
    results = {}
    
    # Test with reduced timesteps for quick comparison
    test_timesteps = 100000
    
    for policy in ["MlpPolicy", "CnnPolicy"]:
        print(f"\n\nTesting {policy}...")
        try:
            model, callback, mean_reward, std_reward = train_dqn_agent(
                policy_type=policy,
                total_timesteps=test_timesteps,
                experiment_name=f"policy_comparison_{policy}",
                save_model=False
            )
            results[policy] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'final_training_reward': callback.rewards[-1] if callback.rewards else 0
            }
        except Exception as e:
            print(f"Error with {policy}: {e}")
            results[policy] = {'error': str(e)}
    
    # Print comparison
    print("\n\n" + "="*80)
    print("POLICY COMPARISON RESULTS")
    print("="*80)
    for policy, result in results.items():
        if 'error' not in result:
            print(f"\n{policy}:")
            print(f"  Evaluation Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"  Final Training Reward: {result['final_training_reward']:.2f}")
        else:
            print(f"\n{policy}: Failed - {result['error']}")
    
    return results


def hyperparameter_tuning_experiments(experiments, member_name):
    """
    Run multiple experiments with different hyperparameter configurations
    Returns a formatted table of results
    """
    
    # Define 10 different hyperparameter configurations    
    results = []
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING EXPERIMENTS")
    print("="*80)
    
    # Run each experiment
    for i, exp in enumerate(experiments, 1):
        print(f"\n\nRunning {exp['name']} ({i}/10)...")
        
        try:
            model, callback, mean_reward, std_reward = train_dqn_agent(
                policy_type="CnnPolicy",
                total_timesteps=500000,
                learning_rate=exp['lr'],
                gamma=exp['gamma'],
                batch_size=exp['batch_size'],
                exploration_initial_eps=exp['eps_start'],
                exploration_final_eps=exp['eps_end'],
                exploration_fraction=exp['exploration_fraction'],
                experiment_name=exp['name']
            )
            
            # Collect results
            result = {
                'experiment': exp['name'],
                'lr': exp['lr'],
                'gamma': exp['gamma'],
                'batch_size': exp['batch_size'],
                'eps_start': exp['eps_start'],
                'eps_end': exp['eps_end'],
                'exploration_fraction': exp['exploration_fraction'],
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'behavior': analyze_behavior(callback, mean_reward)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error in {exp['name']}: {e}")
            results.append({
                'experiment': exp['name'],
                'error': str(e),
                'behavior': 'Failed'
            })
    
    # Print results table
    print_results_table(results)
    
    # Save results to file
    save_results(results, exp['name'], member_name)
    
    return results


def analyze_behavior(callback, mean_reward):
    """
    Analyze training behavior based on metrics
    """
    if len(callback.rewards) < 2:
        return "Insufficient data"
    
    # Check if learning is happening
    initial_reward = np.mean(callback.rewards[:len(callback.rewards)//4])
    final_reward = np.mean(callback.rewards[-len(callback.rewards)//4:])
    improvement = final_reward - initial_reward
    
    # Check stability
    recent_rewards = callback.rewards[-10:]
    stability = np.std(recent_rewards) if len(recent_rewards) > 0 else float('inf')
    
    behavior = []
    
    if improvement > 5:
        behavior.append("Strong learning")
    elif improvement > 0:
        behavior.append("Gradual improvement")
    else:
        behavior.append("No improvement")
    
    if stability < 5:
        behavior.append("stable")
    elif stability < 15:
        behavior.append("moderately stable")
    else:
        behavior.append("unstable")
    
    if mean_reward > 20:
        behavior.append("good performance")
    elif mean_reward > 10:
        behavior.append("moderate performance")
    else:
        behavior.append("poor performance")
    
    return ", ".join(behavior)


def print_results_table(results):
    """
    Print formatted results table
    """
    print("\n\n" + "="*120)
    print("HYPERPARAMETER TUNING RESULTS TABLE")
    print("="*120)
    print(f"{'Experiment':<25} {'LR':<12} {'Gamma':<8} {'Batch':<8} {'Eps Start':<10} {'Eps End':<10} {'Expl Frac':<10} {'Mean Reward':<15} {'Behavior':<30}")
    print("-"*120)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['experiment']:<25} "
                  f"{result['lr']:<12.2e} "
                  f"{result['gamma']:<8.3f} "
                  f"{result['batch_size']:<8} "
                  f"{result['eps_start']:<10.2f} "
                  f"{result['eps_end']:<10.3f} "
                  f"{result['exploration_fraction']:<10.2f} "
                  f"{result['mean_reward']:<15.2f} "
                  f"{result['behavior']:<30}")
        else:
            print(f"{result['experiment']:<25} ERROR: {result['error']}")
    
    print("="*120)


def save_results(results, exp_name, member_name):
    """
    Save results to JSON and formatted text file
    """
    # Create the member-specific results directory if it doesn't exist
    save_dir = os.path.join("results", member_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save as JSON
    with open(f'results/{member_name}/hyperparameter_results_{exp_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as formatted text
    with open(f'results/{member_name}/hyperparameter_results_{exp_name}.txt', 'w') as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("="*120 + "\n\n")
        f.write(f"MEMBER NAME: {member_name}\n\n")
        f.write(f"{'Experiment':<25} | {'Hyperparameters':<80} | {'Behavior':<50}\n")
        f.write("-"*160 + "\n")
        
        for result in results:
            if 'error' not in result:
                hyperparams = f"lr={result['lr']:.2e}, gamma={result['gamma']:.3f}, batch={result['batch_size']}, eps_start={result['eps_start']:.2f}, eps_end={result['eps_end']:.3f}, expl_frac={result['exploration_fraction']:.2f}"
                behavior = f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f} | {result['behavior']}"
                f.write(f"{result['experiment']:<25} | {hyperparams:<80} | {behavior:<50}\n")
            else:
                f.write(f"{result['experiment']:<25} | ERROR: {result['error']}\n")
    
    print(f"\nResults saved to:")
    print(f"  - results/{member_name}/hyperparameter_results_{exp_name}.json")
    print(f"  - results/{member_name}/hyperparameter_results_{exp_name}.txt")


experiments = [
    {
        'name': 'Experiment_1_Baseline',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_2_HighLR',
        'lr': 5e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_3_LowLR',
        'lr': 1e-5,
        'gamma': 0.99,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_4_HighGamma',
        'lr': 1e-4,
        'gamma': 0.995,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_5_LowGamma',
        'lr': 1e-4,
        'gamma': 0.95,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_6_LargeBatch',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_7_SmallBatch',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 16,
        'eps_start': 1.0,
        'eps_end': 0.05,
        'exploration_fraction': 0.1
    },
    {
        'name': 'Experiment_8_SlowExploration',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.1,
        'exploration_fraction': 0.3
    },
    {
        'name': 'Experiment_9_FastExploration',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'exploration_fraction': 0.05
    },
    {
        'name': 'Experiment_10_Aggressive',
        'lr': 1e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'exploration_fraction': 0.05
    }
]


def main():
    """
    Main function to run training experiments
    """
    print("\n" + "="*80)
    print("DQN ATARI TRAINING SCRIPT")
    print("Environment: Breakout-v5")
    print("="*80)
    
    # Option 1: Compare policies (optional)
    print("\n\nOption 1: Compare MLP vs CNN policies")
    print("This will run quick tests to compare policy performance")
    # Uncomment to run: 
    # compare_policies()
    
    # Option 2: Run single training with best hyperparameters
    print("\n\nOption 2: Train single model with specified hyperparameters")
    # model, callback, mean_reward, std_reward = train_dqn_agent(
    #     policy_type="CnnPolicy",
    #     total_timesteps=1000000,
    #     learning_rate=1e-4,
    #     gamma=0.99,
    #     batch_size=32,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.05,
    #     exploration_fraction=0.1,
    #     experiment_name="final_model"
    # )
    
    # Save as the primary model
    # model.save("models/dqn_model.zip")
    # print("\nPrimary model saved as: models/dqn_model.zip")
    
    # Option 3: Run hyperparameter tuning experiments
    print("\n\nOption 3: Run all 10 hyperparameter tuning experiments")
    hyperparameter_tuning_experiments(experiments, "Michael")
    
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nTo run hyperparameter experiments, uncomment the line in main()")
    print("To compare policies, uncomment the compare_policies() call")


if __name__ == "__main__":
    main()