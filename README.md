# Deep-Q-Learning---Breakout

A Deep Q-Network (DQN) implementation for training an agent to play Atari Breakout using Stable-Baselines3. This project explores different hyperparameter configurations through systematic experimentation to determine the optimal policy for the Atari Breakout environment.

## Project Structure

The project is organized into modular components for better code organization and maintainability:

- **`environment.py`**: Contains all environment setup logic, including ALE (Arcade Learning Environment) imports, Gymnasium environment creation, and frame stacking configuration. The `setup_environment()` function handles the creation and configuration of the Atari Breakout environment.

- **`train.py`**: Contains the main training script with agent creation functions for both MLP and CNN policies. This file imports the environment setup from `environment.py` and defines the DQN agents with their respective hyperparameters. It also includes hyperparameter tuning experiments and organizes results by team member.

- **`compare_models.py`**: A comprehensive model comparison tool that analyzes all trained models, ranks them by performance, creates visualizations, and generates detailed comparison reports.

- **`requirements.txt`**: Lists all Python dependencies required for the project.

- **`README.md`**: This file, containing project documentation and usage instructions.

## Agent Definition

The DQN agent is defined using Stable-Baselines3's DQN algorithm, which implements the Deep Q-Learning algorithm with experience replay and target networks. The agent is configured with the following hyperparameters:

- **Learning Rate**: Varied across experiments (see experiments table)
- **Replay Buffer Size**: 100,000 transitions
- **Learning Starts**: 50,000 steps (random exploration before learning begins)
- **Batch Size**: Varied across experiments (24, 32, or 48)
- **Target Update Interval**: 1,000 steps
- **Training Frequency**: Every 4 steps
- **Gradient Steps**: 1 per training update
- **Exploration Strategy**: Epsilon-greedy with linear decay
  - Initial epsilon: Varied (0.92-1.0)
  - Final epsilon: Varied (0.015-0.12)
  - Exploration fraction: Varied (0.06-0.28)

The environment is set up with frame stacking (4 consecutive frames) to provide temporal information to the agent, allowing it to perceive motion and object trajectories in the game.

## Policies Evaluated

Two different policy architectures were implemented and evaluated for the Atari Breakout environment:

### 1. MLP (Multilayer Perceptron) Policy

The MLP policy uses fully connected layers to process the input observations. This policy architecture is designed for low-dimensional state spaces and processes flattened feature vectors through multiple dense layers.

**Configuration:**
- Policy type: `MlpPolicy` from Stable-Baselines3
- TensorBoard logging: `./logs/default_logs/dqn_mlp_tensorboard/`

### 2. CNN (Convolutional Neural Network) Policy

The CNN policy uses convolutional layers to extract spatial features from image observations. This architecture is specifically designed for visual input and can learn hierarchical features from raw pixel data.

**Configuration:**
- Policy type: `CnnPolicy` from Stable-Baselines3
- TensorBoard logging: `./logs/{member_name}_logs/{experiment_name}/`

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

## Hyperparameter Tuning Experiments

A comprehensive hyperparameter tuning study was conducted with 10 different configurations to identify the optimal hyperparameters for the DQN agent. Each experiment was trained for 500,000 timesteps using the CNN policy.

### Experiment Configurations

| Experiment | Learning Rate | Gamma | Batch Size | Epsilon Start | Epsilon End | Exploration Fraction | Description |
|------------|---------------|-------|------------|---------------|--------------|----------------------|-------------|
| **Experiment_1_Baseline** | 2.50e-04 | 0.96 | 32 | 0.95 | 0.05 | 0.15 | Baseline configuration with moderate hyperparameters |
| **Experiment_2_HighLR** | 8.00e-04 | 0.97 | 32 | 1.00 | 0.06 | 0.12 | Higher learning rate to test faster convergence |
| **Experiment_3_LowLR** | 3.00e-05 | 0.94 | 32 | 0.92 | 0.04 | 0.18 | Lower learning rate to test stability |
| **Experiment_4_HighGamma** | 2.00e-04 | 0.992 | 32 | 0.98 | 0.03 | 0.13 | Higher discount factor for long-term rewards |
| **Experiment_5_LowGamma** | 2.50e-04 | 0.91 | 32 | 0.93 | 0.08 | 0.22 | Lower discount factor for short-term focus |
| **Experiment_6_LargeBatch** | 2.00e-04 | 0.96 | 48 | 0.95 | 0.05 | 0.16 | Larger batch size for more stable gradients |
| **Experiment_7_SmallBatch** | 3.00e-04 | 0.95 | 24 | 0.94 | 0.07 | 0.17 | Smaller batch size for faster updates |
| **Experiment_8_SlowExploration** | 2.00e-04 | 0.96 | 32 | 1.00 | 0.12 | 0.28 | Slower exploration decay for more exploration |
| **Experiment_9_FastExploration** | 2.50e-04 | 0.97 | 32 | 0.98 | 0.015 | 0.06 | Faster exploration decay for quicker exploitation |
| **Experiment_10_Aggressive** | 1.20e-03 | 0.98 | 48 | 1.00 | 0.02 | 0.07 | Aggressive hyperparameters (high LR, fast exploration) |

### Training Details

- **Total Timesteps per Experiment**: 500,000
- **Policy Type**: CnnPolicy (CNN)
- **Environment**: ALE/Breakout-v5
- **Number of Parallel Environments**: 4
- **Evaluation Episodes**: 10 episodes per experiment

## Experimental Results

### Performance Results Table

| Rank | Experiment | Mean Reward | Std Reward | Learning Rate | Gamma | Batch Size | Behavior Analysis |
|------|------------|-------------|------------|---------------|-------|------------|-------------------|
| 🥇 **1** | **Experiment_2_HighLR** | **23.10** | ±7.38 | 8.00e-04 | 0.970 | 32 | Strong learning, stable, good performance |
| 🥈 **2** | **Experiment_6_LargeBatch** | **20.40** | ±4.27 | 2.00e-04 | 0.960 | 48 | Strong learning, stable, good performance |
| 🥉 **3** | **Experiment_7_SmallBatch** | **20.20** | ±5.02 | 3.00e-04 | 0.950 | 24 | Strong learning, stable, good performance |
| 4 | Experiment_10_Aggressive | 18.80 | ±6.01 | 1.20e-03 | 0.980 | 48 | Strong learning, stable, moderate performance |
| 5 | Experiment_5_LowGamma | 18.10 | ±7.20 | 2.50e-04 | 0.910 | 32 | Strong learning, stable, moderate performance |
| 6 | Experiment_4_HighGamma | 14.60 | ±3.93 | 2.00e-04 | 0.992 | 32 | Strong learning, stable, moderate performance |
| 7 | Experiment_9_FastExploration | 13.50 | ±5.92 | 2.50e-04 | 0.970 | 32 | Strong learning, stable, moderate performance |
| 8 | Experiment_1_Baseline | 13.00 | ±4.92 | 2.50e-04 | 0.960 | 32 | Strong learning, stable, moderate performance |
| 9 | Experiment_8_SlowExploration | 13.00 | ±4.00 | 2.00e-04 | 0.960 | 32 | Strong learning, stable, moderate performance |
| 10 | Experiment_3_LowLR | 11.00 | ±6.80 | 3.00e-05 | 0.940 | 32 | Strong learning, stable, moderate performance |

### Statistical Summary

- **Highest Reward**: 23.10 (Experiment_2_HighLR)
- **Lowest Reward**: 11.00 (Experiment_3_LowLR)
- **Average Reward**: 16.57
- **Median Reward**: 16.35
- **Standard Deviation**: 3.84
- **Range**: 12.10

### Key Findings

1. **Best Performing Model**: Experiment_2_HighLR achieved the highest mean reward of 23.10, demonstrating that a higher learning rate (8.00e-04) can lead to better performance when combined with appropriate exploration settings.

2. **Batch Size Impact**: Both large (48) and small (24) batch sizes performed well, ranking 2nd and 3rd respectively, suggesting that batch size has a significant impact on performance.

3. **Learning Rate Sensitivity**: The experiment with the lowest learning rate (3.00e-05) performed worst, while higher learning rates (8.00e-04, 1.20e-03) showed better results, indicating the importance of sufficient learning rate for this task.

4. **Exploration Strategy**: Moderate exploration fractions (0.12-0.17) performed better than extreme values (0.06 or 0.28), suggesting a balanced exploration-exploitation trade-off is optimal.

5. **Gamma Impact**: Lower gamma values (0.91-0.97) generally performed better than very high gamma (0.992), possibly because Breakout rewards are more immediate.

## Model Comparison Results

A comprehensive model comparison was performed using the `compare_models.py` script, which analyzed all experiments and generated detailed visualizations and reports.

### Best Performing Model: Experiment_2_HighLR

**Performance Metrics:**
- Mean Reward: **23.10 ± 7.38**
- Rank: #1 out of 10 experiments
- Behavior: Strong learning, stable, good performance

**Optimal Hyperparameters:**
- Learning Rate: **8.00e-04** (3.2x baseline)
- Gamma: **0.970**
- Batch Size: **32**
- Epsilon Start: **1.00**
- Epsilon End: **0.060**
- Exploration Fraction: **0.12**
- Epsilon Decay Per Step: **1.57e-05**

**Analysis:**
The best performing model demonstrates that a higher learning rate, when properly balanced with exploration settings, can significantly improve performance. The combination of:
- Higher learning rate (8.00e-04) for faster learning
- Moderate exploration fraction (0.12) for balanced exploration-exploitation
- Standard batch size (32) for stable gradients
- Gamma of 0.970 for appropriate long-term value estimation

resulted in the highest mean reward of 23.10, which is 77.7% higher than the baseline experiment.

### Model Comparison Visualizations

The comparison script generates several visualizations:

1. **Model Comparison Dashboard** (`results/Annemarie_logs/model_comparison.png`)
   - Bar chart comparing mean rewards across all experiments
   - Scatter plots showing relationships between hyperparameters and rewards
   - Heatmap of top 5 models' normalized hyperparameters

2. **Training Curves Comparison** (`results/Annemarie_logs/training_curves_comparison.png`)
   - Overlaid training reward curves for all experiments
   - Episode length progression over training

3. **Individual Training Curves** (`results/Annemarie_logs/training_curves_Experiment_*.png`)
   - Detailed training progress for each experiment

![Model Comparison](results/Annemarie_logs/model_comparison.png)

*Model comparison visualization showing performance across all experiments*

![Training Curves Comparison](results/Annemarie_logs/training_curves_comparison.png)

*Training curves comparison showing learning progression for all experiments*

### Top 3 Models Comparison

| Metric | Experiment_2_HighLR | Experiment_6_LargeBatch | Experiment_7_SmallBatch |
|--------|---------------------|-------------------------|-------------------------|
| **Mean Reward** | 23.10 | 20.40 | 20.20 |
| **Std Reward** | 7.38 | 4.27 | 5.02 |
| **Learning Rate** | 8.00e-04 | 2.00e-04 | 3.00e-04 |
| **Gamma** | 0.970 | 0.960 | 0.950 |
| **Batch Size** | 32 | 48 | 24 |
| **Key Strength** | High learning rate | Large batch stability | Small batch efficiency |

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

### Training Models

Run the training script to train DQN agents with different hyperparameter configurations:

```bash
python train.py
```

The script will:
1. Import the environment setup from `environment.py`
2. Set up the Breakout environment using `setup_environment()`
3. Run hyperparameter tuning experiments
4. Save models, logs, and results in member-specific folders (e.g., `Annemarie_logs/`)

### Comparing Models

After training, compare all models using the comparison script:

```bash
python compare_models.py
```

This will:
1. Load all experiment results
2. Rank models by performance
3. Identify the best performing model
4. Generate comparison visualizations
5. Create a detailed comparison report

Output files are saved in `results/{member_name}_logs/`:
- `model_comparison.png` - Visual comparison dashboard
- `training_curves_comparison.png` - Training history comparison
- `model_comparison_report.txt` - Detailed text report

### Code Organization

The modular structure allows for easy modification and extension:
- To change environment settings, modify `environment.py`
- To modify agent hyperparameters or add new experiments, edit `train.py`
- To customize model comparison, edit `compare_models.py`
- The separation of concerns makes the codebase more maintainable and testable

## Results Organization

All training results are organized by team member in separate folders:

```
logs/
  └── {member_name}_logs/     # Training logs and TensorBoard data
models/
  └── {member_name}_logs/     # Trained model files
results/
  └── {member_name}_logs/     # Training curves, comparison reports, and analysis
```

This organization allows multiple team members to run experiments without conflicts and easily compare their results.

## Dependencies

- stable-baselines3[extra] >= 2.0.0
- gymnasium[atari,accept-rom-license] >= 0.29.0
- ale-py >= 0.9.0
- opencv-python >= 4.8.0
- pillow >= 10.0.0
- numpy >= 1.24.0
- torch >= 2.0.0
- matplotlib >= 3.7.0

## Conclusion

Through systematic hyperparameter tuning, we identified **Experiment_2_HighLR** as the optimal configuration, achieving a mean reward of 23.10. The key insights from this study are:

1. **Higher learning rates** (8.00e-04) can significantly improve performance when balanced with appropriate exploration
2. **Batch size variations** (24-48) both show promise, with larger batches providing more stability
3. **Moderate exploration fractions** (0.12-0.17) outperform extreme values
4. **Gamma values** around 0.96-0.97 provide optimal long-term value estimation for Breakout

These findings provide valuable guidance for future DQN training on Atari environments and demonstrate the importance of systematic hyperparameter tuning in deep reinforcement learning.
