import os
# Import ale-py FIRST to register Atari environments before gymnasium is used
# This ensures the ALE namespace is available
try:
    import ale_py
    import ale_py.roms
except ImportError:
    pass

# Now import gymnasium and stable-baselines3
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy


def create_mlp_agent(env, learning_rate=1e-4, buffer_size=100000, learning_starts=1000):
    """
    Create a DQN agent with MLP (Multilayer Perceptron) policy.
    
    Args:
        env: The training environment
        learning_rate: Learning rate for the optimizer
        buffer_size: Size of the replay buffer
        learning_starts: Number of steps before learning starts
    
    Returns:
        DQN agent with MLPPolicy
    """
    agent = DQN(
        policy=MlpPolicy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/dqn_mlp_tensorboard/"
    )
    return agent


def create_cnn_agent(env, learning_rate=1e-4, buffer_size=100000, learning_starts=1000):
    """
    Create a DQN agent with CNN (Convolutional Neural Network) policy.
    
    Args:
        env: The training environment
        learning_rate: Learning rate for the optimizer
        buffer_size: Size of the replay buffer
        learning_starts: Number of steps before learning starts
    
    Returns:
        DQN agent with CNNPolicy
    """
    agent = DQN(
        policy=CnnPolicy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/dqn_cnn_tensorboard/"
    )
    return agent

def setup_environment(env_id="ALE/Breakout-v5", n_envs=1, n_stack=4):
    """
    Setup the Atari environment with frame stacking.
    
    Args:
        env_id: The environment ID (default: ALE/Breakout-v5)
        n_envs: Number of parallel environments
        n_stack: Number of frames to stack
    
    Returns:
        Vectorized and frame-stacked environment
    """
    # Create the Atari environment
    # make_atari_env will handle the preprocessing automatically
    try:
        env = make_atari_env(env_id, n_envs=n_envs, seed=0)
    except Exception as e:
        # Fallback: try creating environment directly
        print(f"Warning: make_atari_env failed: {e}")
        print("Attempting direct gymnasium.make...")
        def make_env():
            return gym.make(env_id)
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Stack frames to give the agent temporal information
    env = VecFrameStack(env, n_stack=n_stack)
    
    return env

def main():
    """
    Main function to define and compare DQN agents with different policies.
    """
    print("=" * 60)
    print("DQN Agent Definition for Atari Breakout")
    print("=" * 60)
    
    # Setup environment
    print("\n[1/3] Setting up Breakout environment...")
    env = setup_environment()
    print(f"Environment created: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create MLP agent
    print("\n[2/3] Creating DQN agent with MLPPolicy...")
    mlp_agent = create_mlp_agent(env)
    print("MLP Agent created successfully!")
    print(f"Policy: {mlp_agent.policy}")
    print(f"Policy class: {mlp_agent.policy.__class__.__name__}")
    
    # Create CNN agent
    print("\n[3/3] Creating DQN agent with CNNPolicy...")
    cnn_agent = create_cnn_agent(env)
    print("CNN Agent created successfully!")
    print(f"Policy: {cnn_agent.policy}")
    print(f"Policy class: {cnn_agent.policy.__class__.__name__}")
    
    print("\n" + "=" * 60)
    print("Agent Definition Complete!")
    print("=" * 60)
    
    # Policy Recommendation
    print("\n" + "=" * 60)
    print("POLICY RECOMMENDATION FOR ATARI BREAKOUT")
    print("=" * 60)
    print("\n✅ BEST POLICY: CNN (Convolutional Neural Network) Policy")
    print("\nWhy CNN Policy is Best for Atari Breakout:")
    print("  • Atari observations are IMAGE-BASED (210×160×3 RGB frames)")
    print("  • CNNs excel at extracting spatial features from images")
    print("  • CNNs can learn patterns like:")
    print("    - Ball position and trajectory")
    print("    - Paddle position")
    print("    - Brick locations")
    print("    - Spatial relationships between game elements")
    print("  • All successful DQN implementations for Atari use CNN policies")
    print("  • CNN policies are the standard in Deep RL for visual tasks")
    print("\n❌ MLP Policy is NOT Recommended:")
    print("  • MLP policies are designed for LOW-DIMENSIONAL state spaces")
    print("  • Flattening images (210×160×3 = 100,800 values) loses spatial structure")
    print("  • MLP cannot effectively learn spatial patterns from pixels")
    print("  • Would require massive networks and perform poorly")
    print("\n" + "=" * 60)
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()

