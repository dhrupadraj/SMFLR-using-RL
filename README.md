# ⚡ Spiking Neural Network for D2D Power Control Optimization

This repository implements a **Spiking Neural Network (SNN)**-based **Reinforcement Learning (RL)** agent that optimizes **Device-to-Device (D2D)** power allocation and interference management within a cellular communication environment.  
The model combines *bio-inspired encoding*, *population-based spiking representations*, and *deep policy optimization* to simulate intelligent wireless communication behavior.

---

## 🧠 Overview

The system models a **cellular network** with:
- Multiple **Cellular Users (CUs)**
- Multiple **D2D transmitter-receiver pairs**
- Interference-aware power allocation
- Logarithmic reward formulation for joint SINR optimization

The RL agent uses a **Spiking Actor Network**, trained with population-coded spikes to select optimal D2D transmission actions under variable channel and interference conditions.

---

## ⚙️ Features

✅ Population-coded spike encoding for observation space  
✅ Spiking MLP policy network with trainable parameters  
✅ Reward based on CU and D2D SINR performance  
✅ Replay buffer with policy gradient update  
✅ Matplotlib visualization for training reward evolution  

---

## 🧩 Components

| Component | Description |
|------------|--------------|
| `PopSpikeEncoderRegularSpike` | Converts observations into spike trains via Gaussian population coding |
| `SpikeMLP` | Fully connected spiking neural network processing encoded spikes |
| `PopSpikeDecoder` | Decodes spike populations into continuous-valued actions |
| `SpikingActor` | End-to-end spiking policy combining encoder, SNN core, and decoder |
| `Channel` | Simulated D2D-CU wireless environment with path loss, SINR, and interference models |
| `Agent` | RL agent implementing experience replay and policy updates |

---

## 🚀 How to Run

Clone the repository and install dependencies:

```bash
git clone https://github.com/dhrupadraj/Spiking_D2D_Channel_RL.git
cd Spiking_D2D_Channel_RL
pip install -r requirements.txt
