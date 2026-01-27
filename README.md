# L2M-AID: Autonomous Cyber-Physical Defense Reproduction

This repository contains the reproduction and research extension of the paper **"L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning"**.

## Project Structure

*   `src/envs`: Gymnasium environments for SWaT physics and network simulation.
*   `src/agents`:
    *   `blue/`: L2M-AID agents (Orchestrator, Tactical Squad).
    *   `red/`: Adversarial agents (Scripted and Learning/SAC).
*   `src/modules`: Core AI modules (LLM Client, Embedding Generators).
*   `configs/`: Hyperparameters for MAPPO and Env.
*   `scripts/`: Training and Evaluation entry points.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Code

1.  **Train Baseline**: `python scripts/train_selfplay.py --mode curriculum`
2.  **Adversarial Training**: `python scripts/train_selfplay.py --mode adversarial`
