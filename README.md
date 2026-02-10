# 🤖 Qwen TextWorld ReAct Agent 🤖

Minimal **Vision-Language-Action (VLA)** agent on gridworld using **Recurrent PPO**.

<div align="center">
  <video src="assets/video/unlock_to_unlock.mp4" controls width="40%"></video>
  <video src="assets/video/boss_level.mp4" controls width="40%"></video>
</div>

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/core-not-dumped/miniVLA.git
cd miniVLA
```

### 2. Create environment (recommended)
```bash
conda create -n minivla python=3.13 -y
conda activate minivla
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train with Recurrent PPO
```bash
python train_RecurrentPPO_main.py
```

## Environment

This project uses the BabyAI environments provided by MiniGrid.  
For detailed descriptions of the tasks and environment design, see the official documentation:  
[BabyAI Environments — MiniGrid Documentation](https://minigrid.farama.org/environments/babyai/index.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.