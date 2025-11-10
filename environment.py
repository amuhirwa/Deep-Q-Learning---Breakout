# Import ale-py FIRST to register Atari environments before gymnasium is used
# This ensures the ALE namespace is available
try:
    import ale_py
    import ale_py.roms
except ImportError:
    pass

# Now import gymnasium and stable-baselines3 environment utilities
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


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

