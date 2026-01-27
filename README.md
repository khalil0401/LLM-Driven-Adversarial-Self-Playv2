# LLM-Driven Adversarial Self-Play (L2M-AID V2)

This repository contains the reproduction and research extension of the paper **"L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning"**.

It features an adversarial training loop where a **Blue Team** (Defenders) protects a cyber-physical system (SWaT Water Treatment Plant) against an adaptive **Red Team** (Attackers), utilizing **Large Language Models (LLMs)** for context reasoning and safety evaluation.

## 🚀 Quick Start on Google Colab
The easiest way to run this project is on Google Colab using a T4 GPU.

1.  **Upload** this entire folder to your Google Drive.
2.  **Open** the **[`LLM_Driven_Self_Play.ipynb`](LLM_Driven_Self_Play.ipynb)** notebook in Colab.
3.  **Run All Cells**.

This will automatically install dependencies and start training with a compressed HuggingFace model (`microsoft/Phi-3-mini-4k-instruct`).

## 📂 Project Structure

*   `src/envs`: Gymnasium environments for SWaT physics (`swat_sim.py`) and network simulations (`network_sim.py`).
*   `src/agents`:
    *   `blue/`: L2M-AID agents (Orchestrator, Tactical Squad).
    *   `red/`: Adversarial agents (Reinforcement Learning & Scripted).
*   `src/modules`: Core AI modules (LLM Client, Embedding Generators).
*   `scripts/`: Training and Evaluation entry points.
*   `checkpoints/`: Saved model weights (created during training).
*   `results/`: Training plots and metrics (created during training).

## 💻 Local Installation

```bash
pip install -r requirements.txt
# Optional: For HuggingFace local models
pip install torch transformers accelerate bitsandbytes sentence-transformers
```

## 🛠 Usage

### 1. Adversarial Training (Self-Play)
Run the self-play loop where the Red Agent learns to attack and the Blue Agent learns to defend.

**Basic (Mock LLM - Fast CPU Testing):**
```bash
python scripts/train_selfplay.py --mode adversarial --episodes 100 --provider mock
```

**With HuggingFace (GPU Recommended):**
```bash
python scripts/train_selfplay.py --mode adversarial --episodes 100 --provider huggingface --model microsoft/Phi-3-mini-4k-instruct
```

**With OpenAI (Requires API Key):**
```bash
python scripts/train_selfplay.py --mode adversarial --episodes 100 --provider openai --model gpt-3.5-turbo
```

### 2. Monitor Results
*   **Checkpoints**: Saved to `checkpoints/checkpoint_ep_N.pt` every 20 episodes.
*   **Plots**: `results/training_results.png` is generated at the end of the run, showing PSI (Stability), Rewards, and Losses.

## 🤖 Features
*   **LLM Reward Shaping**: The system asks the LLM to rate the safety of logs (0-1) and adds this to the RL reward signal.
*   **Colab Optimization**: Automatically enables 4-bit quantization and `float16` when running on Colab/T4 GPUs.
*   **Mock Mode**: Simulates LLM responses for rapid testing without GPU/API cost.
