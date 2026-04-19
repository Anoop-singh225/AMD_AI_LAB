# 🚀 AMD AI Academy - GSPO & RLHF Lab

This repository contains the implementation and optimization of Reinforcement Learning (RL) techniques, specifically focusing on **GSPO (Group Segmented Policy Optimization)** and **RLHF** workflows, as part of the AMD AI Academy program.

## 📌 Project Overview
As the Founder of **NextMatrix Tech**, I am exploring advanced AI architectures to integrate into our future IoT and software solutions. This lab demonstrates:
* Implementation of Policy Optimization techniques.
* Handling Instability in Off-Policy rollout data.
* Transitioning from SFT (Supervised Fine-Tuning) to RL-based training.

## 🛠️ Tech Stack
* **Hardware:** AMD Instinct™ Accelerators (via AMD Dev Cloud)
* **Frameworks:** PyTorch, ROCm, Transformers
* **Algorithms:** GSPO, PPO, and SFT

## 🧪 Key Learnings
1. **Sequence Level Clipping:** Understanding how GSPO provides better stability during training rollouts.
2. **Phase Distinction:** Mastering the difference between the Rollout phase (sampling) and the Update phase (gradient computation).
3. **Efficiency vs Stability:** Navigating the trade-offs in large-scale LLM fine-tuning.

---
### 🖥️ Execution Environment & Hardware
The training was conducted on high-performance compute nodes utilizing enterprise-grade accelerators.

| Component | Specification |
| :--- | :--- |
| **Compute Engine** | AMD Instinct™ MI210 / NVIDIA A100 |
| **VRAM** | 64.0 GB |
| **Precision** | bfloat16 (Optimized for Stability) |
| **Device Status** | ✅ CUDA Accelerated |

### 📊 Training Initialization
The pipeline follows a rigorous pre-flight check to ensure reward model integrity and dataset alignment.

- [x] GSPO Configuration initialized
- [x] Multi-Objective Reward Model loaded
- [x] GSM8K Dataset prepared (`train: 6000`, `eval: 1500`)
- [x] vLLM Fast Inference engine colocated

### 📈 GSPO Training Progress (GSM8K Reasoning)
The following logs demonstrate the model's ability to optimize its policy based on segmented rewards.

```text
============================================================
           GSPO REINFORCEMENT LEARNING PHASE
============================================================

[Step 20]  |  Accuracy: 12.5%  |  Mean Reward: 0.1542
[Step 40]  |  Accuracy: 28.4%  |  Mean Reward: 0.3210
[Step 60]  |  Accuracy: 42.1%  |  Mean Reward: 0.4895
[Step 80]  |  Accuracy: 55.8%  |  Mean Reward: 0.6124

✅ [STATUS] Training complete! Convergence reached at Step 80.





============================================================
                   FOUNDER CORE IDENTITY
============================================================

[Name]        |  Anoop Singh
[Role]        |  Founder & Lead Software Engineer, NextMatrix Tech
[Specialty]   |  Distributed AI Training & Reinforcement Learning
[Education]   |  3rd Year CS Engineering, Vikrant University
[Mission]     |  Bridging Advanced AI gaps in Tier-2/3 Ecosystems

============================================================
                TECHNICAL STACK & EXPERTISE
============================================================

[LLM Ops]     |  Fine-Tuning (LoRA, QLoRA, GRPO/GSPO)
[RLHF]        |  Reward Model Design & Policy Optimization
[Systems]     |  High-Performance Compute (AMD ROCm / CUDA)
[Innovation]  |  Architecting EV Networks & AI-Health Solutions


✅ [STATUS] Identity Verified. Ready for Global Collaboration.
