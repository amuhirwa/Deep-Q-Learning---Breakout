"""
Play Script for Trained DQN Model
Loads a trained model and plays Breakout
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
import os
import argparse
import time
from environment import setup_environment
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def list_available_models():
    """List all available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found!")
        return []
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    return sorted(models)


def setup_renderable_environment(env_id="ALE/Breakout-v5", n_stack=4):
    """
    Setup a renderable Atari environment with frame stacking and render_mode='human'
    Creates environment with same preprocessing as training but enables rendering
    
    Args:
        env_id: The environment ID (default: ALE/Breakout-v5)
        n_stack: Number of frames to stack
    
    Returns:
        Vectorized and frame-stacked environment with rendering enabled
    """
    try:
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        
        # Create environment factory with render_mode='human'
        def make_env():
            env = gym.make(env_id, render_mode='human')
            # Apply the same AtariWrapper preprocessing as make_atari_env does
            env = AtariWrapper(env)
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env])
        
        # Stack frames to give the agent temporal information
        env = VecFrameStack(env, n_stack=n_stack)
        
        return env
    except Exception as e:
        print(f"Warning: Failed to create renderable environment: {e}")
        print("Falling back to regular environment setup...")
        # Fallback to regular environment
        return setup_environment(env_id=env_id, n_envs=1, n_stack=n_stack)


def get_render_env(env):
    """
    Extract the renderable environment from a vectorized/frame-stacked environment
    
    Args:
        env: Vectorized environment (may be wrapped with VecFrameStack, VecTransposeImage, etc.)
    
    Returns:
        The underlying gymnasium environment that can be rendered, or None if not found
    """
    current_env = env
    
    # Unwrap through layers: VecFrameStack -> VecTransposeImage -> DummyVecEnv -> AtariWrapper -> gym.Env
    max_unwraps = 10  # Safety limit to prevent infinite loops
    unwrap_count = 0
    
    while unwrap_count < max_unwraps:
        unwrap_count += 1
        
        # Check if we've reached a gymnasium environment with render method
        if hasattr(current_env, 'render') and not hasattr(current_env, 'venv') and not hasattr(current_env, 'envs'):
            return current_env
        
        # Unwrap through different wrapper types
        if hasattr(current_env, 'venv'):
            # VecFrameStack or VecTransposeImage have .venv
            current_env = current_env.venv
        elif hasattr(current_env, 'envs') and len(current_env.envs) > 0:
            # DummyVecEnv has .envs list
            current_env = current_env.envs[0]
        elif hasattr(current_env, 'env'):
            # Generic wrapper (like AtariWrapper) has .env
            current_env = current_env.env
        else:
            # Can't unwrap further
            break
    
    # Final check - if it has render method, use it
    if hasattr(current_env, 'render'):
        return current_env
    
    return None


def play_episode(model, env, render=False, verbose=True):
    """
    Play a single episode with the trained model
    
    Args:
        model: Trained DQN model
        env: Environment to play in
        render: Whether to render the environment
        verbose: Whether to print episode statistics
    
    Returns:
        total_reward: Total reward for the episode
        episode_length: Length of the episode
    """
    # Get renderable environment if rendering is enabled
    render_env = None
    if render:
        render_env = get_render_env(env)
        if render_env is None:
            if verbose:
                print("Warning: Could not extract renderable environment. Rendering disabled.")
            render = False
        else:
            if verbose:
                print(f"Rendering enabled. Render environment type: {type(render_env)}")
                print("A game window should appear. If you don't see it, check if it's behind other windows.")
    
    # For vectorized environments, reset() returns only the observation
    obs = env.reset()
    total_reward = 0
    episode_length = 0
    done = False
    
    # Continue until episode is done
    while not done:
        # Get action from the model (deterministic=True for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment (vectorized env returns: obs, reward, done, info)
        obs, reward, done, info = env.step(action)
        
        # Extract scalar values from arrays (vectorized env returns arrays)
        # For n_envs=1, arrays have shape (1,) so we extract the first element
        if isinstance(reward, np.ndarray):
            episode_reward = float(reward[0])
        else:
            episode_reward = float(reward)
        
        if isinstance(done, np.ndarray):
            done = bool(done[0])
        
        total_reward += episode_reward
        episode_length += 1
        
        # Render the environment
        if render and render_env is not None:
            try:
                # Continue unwrapping to get to the base ALE environment if we're at AtariWrapper
                actual_render_env = render_env
                # Keep unwrapping until we reach the base environment
                while hasattr(actual_render_env, 'env'):
                    next_env = actual_render_env.env
                    # Stop if we'd loop back or if next_env is the same
                    if next_env == actual_render_env:
                        break
                    actual_render_env = next_env
                
                # Call render on the actual environment
                # For ALE with render_mode='human', this should open a window
                render_result = actual_render_env.render()
                
                # For the first frame, give extra time for window to initialize
                if episode_length == 1:
                    time.sleep(0.1)
                else:
                    # Small delay to make rendering visible and allow window to update
                    time.sleep(0.03)  # Increased delay for better visibility
            except Exception as e:
                # If rendering fails, disable it
                if verbose and episode_length == 1:
                    print(f"Warning: Rendering failed: {e}")
                    print("Continuing without rendering...")
                render = False
        
        # Break if episode is done
        if done:
            break
    
    if verbose:
        print(f"Episode Reward: {total_reward:.2f}, Episode Length: {episode_length}")
    
    return total_reward, episode_length


def play_model(model_path, n_episodes=5, render=False, evaluate=True):
    """
    Load and play a trained model
    
    Args:
        model_path: Path to the trained model (.zip file)
        n_episodes: Number of episodes to play
        render: Whether to render the environment
        evaluate: Whether to run formal evaluation
    """
    print("\n" + "="*80)
    print("LOADING TRAINED MODEL")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Setup environment (single environment for playing)
    print("Setting up environment...")
    env = setup_environment(n_envs=1, n_stack=4)
    
    # Load the model
    try:
        model = DQN.load(model_path, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    
    # Print model info
    print(f"\nModel Policy: {model.policy}")
    print(f"Environment: {env}")
    
    # Run formal evaluation if requested
    eval_env = env  # Use same env for evaluation
    if evaluate:
        print("\n" + "="*80)
        print("RUNNING FORMAL EVALUATION")
        print("="*80)
        try:
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=10,
                deterministic=True,
                render=render
            )
            print(f"\nEvaluation Results (10 episodes):")
            print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            print(f"Evaluation error: {e}")
    
    # Create a fresh environment for manual playing to avoid state issues
    print("\nCreating fresh environment for manual play...")
    if render:
        print("Setting up renderable environment with render_mode='human'...")
        play_env = setup_renderable_environment(env_id="ALE/Breakout-v5", n_stack=4)
    else:
        play_env = setup_environment(n_envs=1, n_stack=4)
    
    # Play episodes
    print("\n" + "="*80)
    print(f"PLAYING {n_episodes} EPISODES")
    print("="*80)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(1, n_episodes + 1):
        print(f"\nEpisode {episode}/{n_episodes}:")
        reward, length = play_episode(model, play_env, render=render, verbose=True)
        episode_rewards.append(reward)
        episode_lengths.append(length)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PLAY SUMMARY")
    print("="*80)
    print(f"Episodes Played: {n_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("="*80)
    
    # Clean up
    play_env.close()
    env.close()  # Close the original environment
    print("\nPlay session complete!")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Play a trained DQN model on Breakout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play.py --model models/dqn_model_Favour_Exp1_MidGamma.zip
  python play.py --list
  python play.py --model models/dqn_model_Favour_Exp1_MidGamma.zip --episodes 10 --render
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to the trained model (.zip file)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available models'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=5,
        help='Number of episodes to play (default: 5)'
    )
    
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Render the environment (show gameplay)'
    )
    
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip formal evaluation (faster)'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        models = list_available_models()
        if models:
            print("\nAvailable Models:")
            print("-" * 80)
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
            print("-" * 80)
            print(f"\nTotal: {len(models)} models")
            print("\nUse --model to specify which model to play")
        else:
            print("No models found in models/ directory")
        return
    
    # Get model path
    model_path = args.model
    
    # If no model specified, try to use the first available model
    if model_path is None:
        models = list_available_models()
        if models:
            model_path = os.path.join("models", models[0])
            print(f"No model specified. Using: {model_path}")
        else:
            print("Error: No model specified and no models found!")
            print("Use --list to see available models or --model to specify a model")
            return
    
    # Ensure model path is correct
    if not model_path.startswith("models/"):
        if os.path.exists(model_path):
            pass  # Full path provided
        else:
            model_path = os.path.join("models", model_path)
    
    # Play the model
    play_model(
        model_path=model_path,
        n_episodes=args.episodes,
        render=args.render,
        evaluate=not args.no_eval
    )


if __name__ == "__main__":
    main()

