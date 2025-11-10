import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
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


if __name__ == "__main__":
    main()

