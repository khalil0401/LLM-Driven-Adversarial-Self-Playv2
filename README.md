# L2M-AID V2: LLM-Driven Adversarial Self-Play for Generic CPS Defense

![Training Result](results/training_results.png)

This repository contains the official implementation of **L2M-AID V2**, an autonomous defense framework for Cyber-Physical Systems (CPS). It fuses **Hierarchical Multi-Agent Reinforcement Learning (MARL)** with **Large Language Models (LLMs)** to create adaptive, explainable, and robust defense strategies against evolving adversarial threats.

## 🌟 Key Research Capabilities

### 1. Hybrid Environment Engine
The framework operates in two distinct modes to support both theoretical and data-driven research:

*   **Mode A: Physics-Based (LTI)**  
    Uses a mathematically generalized **Linear Time-Invariant** physics engine ($dx/dt = -Ax + Bu + Noise$) to simulate generic CPS dynamics (Power Grids, Water Plants).
    
*   **Mode B: Data-Driven IoT (TON_IoT)**  
    Directly ingests **TON_IoT Dataset** (CSV/Parquet) to replay real-world telemetry sequences. The environment transforms static dataset rows into dynamic states, allowing agents to interact with real traffic flows.

### 2. Adversarial Self-Play (Red vs. Blue)
-   **Red Team (Attacker)**: Uses **Entropy-Regularized REINFORCE** to learn optimal attack sequences (DoS, Integrity Spoofing, Actuator MitM).
-   **Blue Team (Defender)**: Uses **MAPPO (Multi-Agent PPO)** constrained by semantic safety guidance.
-   **Dynamics**: Zero-sum game structure driving continuous adaptation (Arms Race).

### 3. Generative Explainability
-   **Semantic Root Cause Analysis**: An LLM-based agent monitors the episode trace and generates human-readable reports explaining *unstable states* and *agent decisions*.
-   **Safety Scoring**: Real-time semantic evaluation of system logs to shape RL rewards ($R = PSI + 0.5 \cdot S_{LLM}$).

## 📂 Project Structure

*   `src/envs`:
    *   `generic_cps.py`: The LTI Physics Engine.
    *   `data_driven_cps.py`: The TON_IoT Dataset Replayer.
    *   `wrapper.py`: Unified Gym Interface switching between Physics and Data modes.
*   `src/data_loader.py`: Metric extraction and normalization for IoT datasets.
*   `src/agents`:
    *   `blue/orchestrator.py`: Strategic LLM Agent (Brain).
    *   `blue/tactical.py`: Low-level MAPPO Control Agents (Body).
    *   `red/learning.py`: Adversarial Learning Agent.
*   `src/modules`:
    *   `explanation.py`: Generative Post-Hoc Explainer.

## 🚀 Quick Start (Google Colab)

The easiest way to replicate our experiments is using the provided Notebook on a T4 GPU.

### 1. Pure Simulation (Physics Mode)
Run the standard adversarial self-play on the LTI physics engine.

```bash
python scripts/train_selfplay.py --mode adversarial --episodes 500 --provider huggingface --model microsoft/Phi-3-mini-4k-instruct
```

### 2. Data-Driven IoT (TON_IoT Mode)
Run the system on real dataset traces. Upload `Train_Test_Network.csv` to your environment first.

```bash
python scripts/train_selfplay.py --mode adversarial --episodes 500 --dataset "/content/Train_Test_Network.csv" --provider huggingface --model microsoft/Phi-3-mini-4k-instruct
```

### 3. Local Hybrid Execution (Windows/Linux)
If you have the dataset locally (e.g., `data/TON/train_test_network.csv`):

```bash
python scripts/train_selfplay.py --mode adversarial --episodes 500 --dataset "data/TON/train_test_network.csv" --hybrid
```

*Note: If the dataset file is not found, the system will automatically fall back to a synthetic IoT traffic generator.*

## 💻 Local Installation

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Training (Mock LLM - Fast CPU)
python scripts/train_selfplay.py --mode adversarial --episodes 100 --provider mock
```

## 📊 Outputs

*   **Plots**: `results/training_results.png` (Updated live).
*   **Explanations**: `results/explanations.txt` (Generated every 50 episodes).
*   **Checkpoints**: `checkpoints/` (Saved weights).

## 📜 Citation

If you use this code, please cite the original paper:
> *L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning.*
