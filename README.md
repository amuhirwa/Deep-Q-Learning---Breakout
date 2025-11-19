# Deep Q-Learning for Atari Breakout

A Deep Q-Network (DQN) implementation for training an agent to play Atari Breakout using Stable-Baselines3. This project explores different hyperparameter configurations through systematic experimentation to determine optimal training strategies.

**Demo Video**: [Watch Here](https://drive.google.com/file/d/1RhiddOiLw6MOAEnELpubP8qma0tG-r83/view?usp=sharing)

## Table of Contents

- [Project Structure](#project-structure)
- [Team Contributions](#team-contributions)
- [Installation](#installation)
- [Usage](#usage)
- [Agent Definition](#agent-definition)
- [Policy Comparison](#policy-comparison)
- [Hyperparameter Experiments](#hyperparameter-experiments)
- [Key Findings](#key-findings)
- [Conclusion](#conclusion)

## Project Structure

- **`environment.py`**: Environment setup logic (ALE imports, Gymnasium environment creation, frame stacking)
- **`train.py`**: Main training script with agent creation, hyperparameter tuning, and logging
- **`compare_models.py`**: Model comparison tool for analyzing trained models and generating visualizations
- **`play.py`**: Model evaluation script with rendering and command-line interface
- **`requirements.txt`**: Python dependencies

## Team Contributions

| Team Member    | Contribution                                                                                                                                                                                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Anne Marie** | DQN agent definition using Stable Baselines3. Implemented and compared MLPPolicy vs CNNPolicy to determine optimal architecture. Created the `compare_policies()` function.                                                                                                      |
| **Michael**    | Created `train_dqn_agent()` function with comprehensive hyperparameter configuration. Set up training infrastructure including logging, checkpoints, and the `TrainingCallback` class. Developed `plot_training_curves()` visualization. Conducted boundary-testing experiments. |
| **Joan**       | Developed hyperparameter tuning framework with `hyperparameter_tuning_experiments()` function. Implemented `analyze_behavior()` for training analysis and created result tables with `print_results_table()` and `save_results()` functions.                                     |
| **Favour**     | Developed `play.py` script for model evaluation. Implemented model loading, episode playing with rendering, and command-line interface for interactive testing via `play_episode()` and `play_model()` functions.                                                                |

## Installation

```bash
git clone <repository-url>
cd Deep-Q-Learning---Breakout
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
python train.py
```

### Comparing Models

```bash
python compare_models.py
```

### Playing Trained Models

```bash
python play.py --model <path_to_model.zip> --episodes 10 --render
```

## Agent Definition

The DQN agent uses Stable-Baselines3's DQN implementation with experience replay and target networks:

**Fixed Configuration:**

- **Replay Buffer Size**: 100,000 transitions
- **Learning Starts**: 50,000 steps (random exploration before learning)
- **Target Update Interval**: 1,000 steps
- **Training Frequency**: Every 4 steps
- **Gradient Steps**: 1 per training update
- **Frame Stacking**: 4 consecutive frames for temporal information

**Varied Parameters** (see experiment tables):

- Learning Rate (1e-5 to 1e-2)
- Gamma (0.0 to 0.999)
- Batch Size (8 to 64)
- Exploration Strategy (epsilon-greedy with varied decay rates)

## Policy Comparison

### CNN Policy (Chosen) ✅

**Why CNN?**

- Designed for image-based observations (210×160×3 RGB frames)
- Extracts spatial features via convolutional layers
- Learns hierarchical patterns (ball position, paddle location, brick patterns)
- Industry standard for Atari games (proven in DeepMind's original DQN paper)
- Efficient parameter sharing and local connectivity

### MLP Policy (Not Suitable) ❌

**Why Not MLP?**

- Designed for low-dimensional state spaces, not high-dimensional images
- Flattening destroys spatial relationships (100,800 values)
- Cannot effectively learn spatial patterns from pixels
- Poor performance on visual tasks

## Hyperparameter Experiments

Four team members conducted systematic hyperparameter studies totaling 40 experiments across 500,000 timesteps each using CNNPolicy.

---

### Anne Marie's Experiments (Moderate Variations)

**Approach**: Fine-tuning within safe parameter ranges

| Rank | Experiment                   | Mean Reward | Std   | LR       | Gamma | Batch | Exploration (ε: start→end, frac) | Key Insight                                               |
| ---- | ---------------------------- | ----------- | ----- | -------- | ----- | ----- | -------------------------------- | --------------------------------------------------------- |
| 🥇 1 | **Experiment_2_HighLR**      | **23.10**   | ±7.38 | 8.00e-04 | 0.970 | 32    | 1.0→0.06 (0.12)                  | High LR (3.2x baseline) with moderate exploration optimal |
| 🥈 2 | Experiment_6_LargeBatch      | 20.40       | ±4.27 | 2.00e-04 | 0.960 | 48    | 0.95→0.05 (0.16)                 | Large batch provides excellent stability                  |
| 🥉 3 | Experiment_7_SmallBatch      | 20.20       | ±5.02 | 3.00e-04 | 0.950 | 24    | 0.94→0.07 (0.17)                 | Small batch enables fast learning                         |
| 4    | Experiment_10_Aggressive     | 18.80       | ±6.01 | 1.20e-03 | 0.980 | 48    | 1.0→0.02 (0.07)                  | Very high LR needs large batch for stability              |
| 5    | Experiment_5_LowGamma        | 18.10       | ±7.20 | 2.50e-04 | 0.910 | 32    | 0.93→0.08 (0.22)                 | Low gamma suits immediate rewards                         |
| 6    | Experiment_4_HighGamma       | 14.60       | ±3.93 | 2.00e-04 | 0.992 | 32    | 0.98→0.03 (0.13)                 | Very high gamma over-emphasizes distant rewards           |
| 7    | Experiment_9_FastExploration | 13.50       | ±5.92 | 2.50e-04 | 0.970 | 32    | 0.98→0.015 (0.06)                | Quick exploitation limits exploration                     |
| 8    | Experiment_1_Baseline        | 13.00       | ±4.92 | 2.50e-04 | 0.960 | 32    | 0.95→0.05 (0.15)                 | Conservative baseline reference                           |
| 9    | Experiment_8_SlowExploration | 13.00       | ±4.00 | 2.00e-04 | 0.960 | 32    | 1.0→0.12 (0.28)                  | Excessive exploration limits exploitation                 |
| 10   | Experiment_3_LowLR           | 11.00       | ±6.80 | 3.00e-05 | 0.940 | 32    | 0.92→0.04 (0.18)                 | Very low LR severely limits learning                      |

**Key Findings**:

- Optimal LR range: 8.00e-04 to 1.20e-03
- Batch size flexibility: Both 24 and 48 performed well
- Exploration sweet spot: 0.12-0.17 fraction
- Gamma range: 0.95-0.97 optimal for immediate rewards

---

### Michael's Experiments (Boundary Testing)

**Approach**: Extreme parameter values to identify failure modes

| Rank | Experiment                       | Mean Reward | Std   | LR       | Gamma | Batch | Exploration (ε: start→end, frac) | Key Insight                                               |
| ---- | -------------------------------- | ----------- | ----- | -------- | ----- | ----- | -------------------------------- | --------------------------------------------------------- |
| 🥇 1 | **Experiment_10_Aggressive**     | **27.0**    | ±7.62 | 1.00e-03 | 0.950 | 64    | 1.0→0.01 (0.05)                  | **Best overall** - Aggressive settings work when balanced |
| 🥈 2 | Experiment_1_Baseline            | 16.0        | ±1.79 | 1.00e-04 | 0.990 | 32    | 1.0→0.05 (0.10)                  | Solid baseline with excellent stability                   |
| 🥉 3 | Experiment_5_VeryPatient         | 15.2        | ±3.99 | 1.00e-04 | 0.999 | 32    | 1.0→0.05 (0.10)                  | Very high gamma slightly overcomplicates                  |
| 4    | Experiment_9_ExtendedExploration | 15.1        | ±3.81 | 1.00e-04 | 0.990 | 32    | 1.0→0.2 (0.50)                   | Extended exploration (50%) no better than standard        |
| 5    | Experiment_6_TinyBatch           | 12.2        | ±2.89 | 1.00e-04 | 0.990 | 8     | 1.0→0.05 (0.10)                  | Tiny batch functional but suboptimal                      |
| 6    | Experiment_7_PureExploitation    | 11.9        | ±4.53 | 1.00e-04 | 0.990 | 32    | 0.0→0.0 (0.00)                   | Zero exploration drops performance 26%                    |
| 7    | Experiment_8_NoExploration       | 11.7        | ±2.15 | 1.00e-04 | 0.990 | 32    | 0.1→0.01 (0.05)                  | Minimal exploration insufficient                          |
| 8    | Experiment_4_Myopic              | 7.1         | ±3.01 | 1.00e-04 | 0.500 | 32    | 1.0→0.05 (0.10)                  | Gamma 0.5 too myopic even for Breakout                    |
| 9    | Experiment_2_VeryHighLR          | 6.6         | ±2.20 | 1.00e-02 | 0.990 | 32    | 1.0→0.05 (0.10)                  | LR 100x baseline causes catastrophic failure              |
| 10   | Experiment_3_ZeroDiscount        | 3.3         | ±1.85 | 1.00e-04 | 0.000 | 32    | 1.0→0.05 (0.10)                  | Zero gamma completely fails - no sequential learning      |

**Key Findings**:

- **Boundaries Identified**:
  - LR: Effective range 1e-4 to 1e-3; above 1e-2 fails
  - Gamma: Must be > 0.5; optimal 0.95-0.99
  - Exploration: Must explore > 5%
  - Batch: Works from 8 to 64; larger better for high LR
- **Critical**: Gamma is most impactful (range: 3.3 to 27.0 reward)
- **Best combo**: High LR (1e-3) + Large Batch (64) + Fast Exploration (0.05)

---

### Joan's Experiments (Diverse Configurations)

**Approach**: Varied configurations testing multiple parameter combinations

| Rank | Experiment                   | Mean Reward | Std   | LR       | Gamma | Batch | Exploration (ε: start→end, frac) | Key Insight                                         |
| ---- | ---------------------------- | ----------- | ----- | -------- | ----- | ----- | -------------------------------- | --------------------------------------------------- |
| 🥇 1 | **Experiment_10_Aggressive** | **20.1**    | ±5.01 | 1.00e-03 | 0.990 | 64    | 1.0→0.01 (0.08)                  | High LR with large batch achieves strong results    |
| 🥈 2 | Experiment_2_HighLR          | 19.0        | ±4.73 | 7.00e-04 | 0.980 | 32    | 0.95→0.08 (0.15)                 | High LR (7x baseline) effective with standard batch |
| 🥉 3 | Experiment_9_FastExploration | 17.3        | ±5.73 | 2.50e-04 | 0.970 | 32    | 1.0→0.02 (0.05)                  | Fast exploration with moderate LR balanced          |
| 4    | Experiment_4_HighGamma       | 15.9        | ±4.48 | 1.50e-04 | 0.995 | 32    | 1.0→0.03 (0.12)                  | Very high gamma (0.995) decent but not optimal      |
| 5    | Experiment_5_LowGamma        | 15.8        | ±4.60 | 2.00e-04 | 0.900 | 32    | 0.9→0.07 (0.30)                  | Low gamma (0.9) with extended exploration works     |
| 6    | Experiment_6_LargeBatch      | 15.3        | ±3.32 | 1.20e-04 | 0.970 | 64    | 1.0→0.1 (0.18)                   | Low LR limits large batch advantage                 |
| 7    | Experiment_8_SlowExploration | 14.9        | ±3.11 | 1.00e-04 | 0.980 | 32    | 1.0→0.15 (0.35)                  | Very slow exploration (35%) comparable to baseline  |
| 8    | Experiment_1_Baseline        | 14.3        | ±4.36 | 2.00e-04 | 0.950 | 64    | 0.9→0.1 (0.20)                   | Baseline with large batch and slower exploration    |
| 9    | Experiment_7_SmallBatch      | 14.3        | ±3.69 | 3.00e-04 | 0.960 | 16    | 0.95→0.06 (0.14)                 | Small batch (16) works with moderate LR             |
| 10   | Experiment_3_LowLR           | 7.7         | ±2.28 | 5.00e-05 | 0.920 | 64    | 0.98→0.05 (0.25)                 | Very low LR (5e-5) severely limits learning         |

**Key Findings**:

- Confirms high LR (7e-4 to 1e-3) with appropriate batch size is optimal
- Gamma range 0.95-0.99 consistently performs well
- Exploration fraction 0.05-0.20 all viable
- Large batch (64) requires sufficient LR to leverage stability advantage

---

### Favour's Experiments (Exploration Focus)

**Approach**: Testing different exploration strategies and gamma values

| Rank | Experiment                           | Mean Reward | Std   | Description                                                       |
| ---- | ------------------------------------ | ----------- | ----- | ----------------------------------------------------------------- |
| 🥇 1 | **Favour_Exp5_LowGamma_Explorative** | **17.3**    | ±2.72 | High LR (6e-4), low gamma (0.94), slow exploration (0.3 fraction) |
| 2    | Favour_Exp2_HigherLR_Stable          | 13.8        | ±7.95 | Higher LR with standard settings                                  |
| 3    | Favour_Exp6_VeryLargeBatch           | 13.0        | ±4.12 | Very large batch size configuration                               |
| 4    | Favour_Exp10_VeryExplorative         | 10.5        | ±5.18 | Extended exploration testing                                      |
| 5    | Favour_Exp8_MidGamma_SlowExploration | 9.3         | ±4.67 | Mid gamma with slow exploration                                   |
| 6    | Favour_Exp9_AggressiveButLongTerm    | 9.9         | ±2.26 | Aggressive LR with high gamma                                     |
| 7    | Favour_Exp4_ShortHorizon             | 8.3         | ±2.57 | Short-term horizon testing                                        |
| 8    | Favour_Exp3_VeryLowLR                | 6.0         | ±2.37 | Very low learning rate                                            |
| 9    | Favour_Exp1_MidGamma                 | 5.2         | ±2.18 | Mid-range gamma baseline                                          |
| 10   | Favour_Exp7_HighDiscount_SlowLR      | 4.9         | ±1.58 | High discount with slow LR                                        |

**Key Findings**:

- Extended exploration (30% fraction) with moderate gamma (0.94) effective
- High gamma (0.99) across experiments plateaued around 13 reward
- Very low LR underperforms regardless of other settings

---

## Key Findings

### Overall Best Configuration

**Michael's Experiment_10_Aggressive: 27.0 ± 7.62 reward**

- Learning Rate: 1.00e-03
- Gamma: 0.95
- Batch Size: 64
- Exploration: Fast (1.0 → 0.01, fraction 0.05)

### Parameter Insights

**1. Learning Rate**

- **Optimal**: 8e-4 to 1e-3
- **Too Low** (< 5e-5): Severely limits learning
- **Too High** (> 1e-2): Causes instability and failure
- **Key**: Higher LR requires larger batch for stability

**2. Gamma (Discount Factor)**

- **Optimal**: 0.95-0.97 for Breakout
- **Critical Threshold**: < 0.5 prevents sequential learning
- **Too High** (> 0.99): Overcomplicates without benefit
- **Zero**: Complete failure (3.3 reward)

**3. Batch Size**

- **Large (48-64)**: Enables high LR, provides stability
- **Standard (32)**: Versatile across configurations
- **Small (16-24)**: Works but needs careful LR tuning
- **Tiny (8)**: Functional but suboptimal

**4. Exploration**

- **Optimal Fraction**: 0.05-0.20
- **Minimum**: At least 5% essential
- **Extended** (> 30%): No additional benefit
- **Zero**: 25-27% performance drop

**5. Parameter Priority** (Most to Least Impactful)

1. **Learning Rate** - Critical threshold effect
2. **Gamma** - Largest impact range (3.3 to 27.0)
3. **Exploration** - Must explore, but wide viable range
4. **Batch Size** - Flexibility within 16-64 range

### Cross-Team Insights

| Member         | Best Reward | LR   | Gamma | Batch | Approach            |
| -------------- | ----------- | ---- | ----- | ----- | ------------------- |
| **Michael**    | 27.0        | 1e-3 | 0.95  | 64    | Boundary testing    |
| **Anne Marie** | 23.1        | 8e-4 | 0.97  | 32    | Moderate variations |
| **Joan**       | 20.1        | 1e-3 | 0.99  | 64    | Diverse configs     |
| **Favour**     | 17.3        | 6e-4 | 0.94  | 32    | Exploration focus   |

**Key Pattern**: Higher rewards achieved with:

- Higher LR (7e-4 to 1e-3)
- Moderate-low gamma (0.94-0.97)
- Larger batch sizes (32-64)
- Faster exploration (0.05-0.15 fraction)

## Conclusion

### Practical Recommendations

**For Best Performance**:

1. Start with: LR=1e-3, Batch=64, Gamma=0.95, Exploration=0.05
2. If unstable: Reduce LR to 8e-4, keep large batch
3. For stability: Use batch 48-64 with any LR above baseline
4. **Avoid**: Gamma < 0.9, LR > 1e-2, zero exploration, batch < 16

**Value of Multiple Approaches**:

- **Anne Marie**: Demonstrated fine-tuning effectiveness within safe ranges
- **Michael**: Identified critical boundaries and validated aggressive configurations
- **Joan**: Confirmed robustness across diverse parameter combinations
- **Favour**: Showed exploration strategy importance

The combined insights from 40 experiments provide comprehensive guidance for DQN training on Atari Breakout and demonstrate the value of systematic hyperparameter exploration in deep reinforcement learning.

### Results Organization

```
logs/{member_name}_logs/          # TensorBoard logs
models/{member_name}_logs/        # Trained models (.zip)
results/{member_name}_logs/       # Visualizations and reports
```

---

**Dependencies**: stable-baselines3, gymnasium[atari], ale-py, torch, matplotlib, opencv-python

**Environment**: ALE/Breakout-v5 with 4-frame stacking
