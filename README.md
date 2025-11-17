# Deep-Q-Learning---Breakout

A Deep Q-Network (DQN) implementation for training an agent to play Atari Breakout using Stable-Baselines3. This project explores different policy architectures and determines the optimal policy for the Atari Breakout environment.

## Project Structure

The project is organized into modular components for better code organization and maintainability:

- **`environment.py`**: Contains all environment setup logic, including ALE (Arcade Learning Environment) imports, Gymnasium environment creation, and frame stacking configuration. The `setup_environment()` function handles the creation and configuration of the Atari Breakout environment.

- **`train.py`**: Contains the main training script with agent creation functions for both MLP and CNN policies. This file imports the environment setup from `environment.py` and defines the DQN agents with their respective hyperparameters.

- **`requirements.txt`**: Lists all Python dependencies required for the project.

- **`README.md`**: This file, containing project documentation and usage instructions.

## Agent Definition

The DQN agent is defined using Stable-Baselines3's DQN algorithm, which implements the Deep Q-Learning algorithm with experience replay and target networks. The agent is configured with the following hyperparameters:

- **Learning Rate**: 1e-4
- **Replay Buffer Size**: 100,000 transitions
- **Learning Starts**: 1,000 steps (random exploration before learning begins)
- **Batch Size**: 32 samples per gradient update
- **Target Update Interval**: 1,000 steps
- **Training Frequency**: Every 4 steps
- **Gradient Steps**: 1 per training update
- **Exploration Strategy**: Epsilon-greedy with linear decay
  - Initial epsilon: 1.0 (100% random actions)
  - Final epsilon: 0.01 (1% random actions)
  - Exploration fraction: 0.1 (10% of training time)

The environment is set up with frame stacking (4 consecutive frames) to provide temporal information to the agent, allowing it to perceive motion and object trajectories in the game.

## Policies Evaluated

Two different policy architectures were implemented and evaluated for the Atari Breakout environment:

### 1. MLP (Multilayer Perceptron) Policy

The MLP policy uses fully connected layers to process the input observations. This policy architecture is designed for low-dimensional state spaces and processes flattened feature vectors through multiple dense layers.

**Configuration:**
- Policy type: `MlpPolicy` from Stable-Baselines3
- TensorBoard logging: `./logs/dqn_mlp_tensorboard/`

### 2. CNN (Convolutional Neural Network) Policy

The CNN policy uses convolutional layers to extract spatial features from image observations. This architecture is specifically designed for visual input and can learn hierarchical features from raw pixel data.

**Configuration:**
- Policy type: `CnnPolicy` from Stable-Baselines3
- TensorBoard logging: `./logs/dqn_cnn_tensorboard/`

## Best Model & Experiment Results

After running the hyperparameter sweep in `results/Favour/hyperparameter_results_all_experiments_20251115_013535.[json|txt]`, the best-performing checkpoint was **Favour_Exp5_LowGamma_Explorative**. Its model file lives at:

```
models/dqn_model_Favour_Exp5_LowGamma_Explorative.zip
```

| Experiment | Mean Reward | Std Reward | Behavior | Model Path |
|------------|-------------|------------|----------|------------|
| Favour_Exp1_MidGamma | 5.2 | 2.18 | No improvement, stable, poor performance | `models/dqn_model_Favour_Exp1_MidGamma.zip` |
| Favour_Exp2_HigherLR_Stable | 13.8 | 7.95 | No improvement, stable, moderate performance | `models/dqn_model_Favour_Exp2_HigherLR_Stable.zip` |
| Favour_Exp3_VeryLowLR | 6.0 | 2.37 | No improvement, stable, poor performance | `models/dqn_model_Favour_Exp3_VeryLowLR.zip` |
| Favour_Exp4_ShortHorizon | 8.3 | 2.57 | No improvement, stable, poor performance | `models/dqn_model_Favour_Exp4_ShortHorizon.zip` |
| **Favour_Exp5_LowGamma_Explorative** | **17.3** | **2.72** | **No improvement, stable, moderate performance** | **`models/dqn_model_Favour_Exp5_LowGamma_Explorative.zip`** |
| Favour_Exp6_VeryLargeBatch | 13.0 | 4.12 | No improvement, stable, moderate performance | `models/dqn_model_Favour_Exp6_VeryLargeBatch.zip` |
| Favour_Exp7_HighDiscount_SlowLR | 4.9 | 1.58 | No improvement, stable, poor performance | `models/dqn_model_Favour_Exp7_HighDiscount_SlowLR.zip` |
| Favour_Exp8_MidGamma_SlowExploration | 9.3 | 4.67 | No improvement, stable, poor performance | `models/dqn_model_Favour_Exp8_MidGamma_SlowExploration.zip` |
| Favour_Exp9_AggressiveButLongTerm | 9.9 | 2.26 | No improvement, stable, poor performance | `models/dqn_model_Favour_Exp9_AggressiveButLongTerm.zip` |
| Favour_Exp10_VeryExplorative | 10.5 | 5.18 | No improvement, stable, moderate performance | `models/dqn_model_Favour_Exp10_VeryExplorative.zip` |

You can play any model with:

```bash
python play.py --model <path_to_model.zip> --episodes 10 --render
```

## Chosen Policy for Atari Breakout

**CNN (Convolutional Neural Network) Policy** was selected as the optimal policy architecture for the Atari Breakout environment.

### Rationale

The CNN policy is the best-performing choice for this environment due to several key factors:

1. **Image-Based Observations**: Atari Breakout provides observations as RGB image frames (210×160×3 pixels). The CNN architecture is specifically designed to process and extract meaningful features from visual data.

2. **Spatial Feature Extraction**: CNNs excel at learning spatial patterns and relationships in images. For Breakout, this enables the agent to identify:
   - Ball position and trajectory
   - Paddle location
   - Brick positions and patterns
   - Spatial relationships between game elements

3. **Industry Standard**: All successful DQN implementations for Atari games utilize CNN policies. This architecture has been proven effective in the original DeepMind DQN paper and subsequent research.

4. **Efficient Processing**: CNNs process image data more efficiently than fully connected networks by leveraging parameter sharing and local connectivity, making them computationally efficient for visual tasks.

### Why MLP Policy is Not Suitable

The MLP policy is not recommended for Atari Breakout because:

- **Designed for Low-Dimensional Spaces**: MLP policies are intended for environments with low-dimensional state representations, not high-dimensional image data.

- **Loss of Spatial Structure**: Flattening images (210×160×3 = 100,800 values) destroys the spatial relationships that are crucial for understanding the game state.

- **Inefficient Learning**: MLP networks cannot effectively learn spatial patterns from pixel data and would require significantly larger networks to achieve comparable performance, if at all.

- **Poor Performance**: Empirical evidence and theoretical understanding indicate that MLP policies would perform poorly on visual tasks like Atari games.

## Environment Setup

The environment setup is handled in `environment.py`, which encapsulates all environment-related configuration. The project uses the `ALE/Breakout-v5` environment from Gymnasium, which provides the standard Atari Breakout game with:
- Preprocessed observations (grayscale, resized)
- Frame stacking for temporal information (4 consecutive frames by default)
- Standard Atari action space
- Vectorized environment support for parallel training

The `setup_environment()` function in `environment.py` handles:
- ALE namespace registration for Atari environments
- Environment creation with automatic preprocessing
- Frame stacking configuration
- Error handling with fallback mechanisms

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the training script to define and compare the DQN agents:

```bash
python train.py
```

The script will:
1. Import the environment setup from `environment.py`
2. Set up the Breakout environment using `setup_environment()`
3. Create DQN agents with both MLP and CNN policies
4. Display policy recommendations for the environment

### Code Organization

The modular structure allows for easy modification and extension:
- To change environment settings, modify `environment.py`
- To modify agent hyperparameters or add new policies, edit `train.py`
- The separation of concerns makes the codebase more maintainable and testable

## Dependencies

- stable-baselines3[extra] >= 2.0.0
- gymnasium[atari,accept-rom-license] >= 0.29.0
- ale-py >= 0.9.0
- opencv-python >= 4.8.0
- pillow >= 10.0.0
- numpy >= 1.24.0
- torch >= 2.0.0
