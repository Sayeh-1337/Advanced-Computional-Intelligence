# Advanced Computational Intelligence with NEAT-CA, Hypergraph SNN, and STDP

This repository implements a sophisticated AI framework combining NeuroEvolution of Augmenting Topologies (NEAT), Cellular Automata (CA), Variational Autoencoders (VAE), Hypergraph Spiking Neural Networks (SNN), and Spike-Timing-Dependent Plasticity (STDP). The system is designed to simulate complex, adaptive behaviors using brain-inspired computational techniques.

## Project Structure

### Files

- `pong-game-qvsneat.py`: Implements a Pong game where NEAT-CA competes against a Q-learning agent.
- `maze-solver-3.py`: Maze-solving agent using NEAT-CA, Hypergraph SNN, and STDP.
- `uav_main3.py`: UAV control simulation using Hypergraph SNNs with STDP and NEAT-CA.
- `QuantumCryptoAgent.py`: Implements a quantum-assisted portfolio optimization agent using NEAT, SNN, VAE, and Hypergraph layers with PPO reinforcement learning for cryptocurrency trading

### Requirements

- **Python Libraries**:
  - `pygame`: For game rendering and control.
  - `neat-python`: For NEAT algorithm implementation.
  - `numpy`: For numerical computations.
  - `networkx`: For creating and managing hypergraphs.
  - `torch`: For neural network operations and GPU support.
  - `snntorch`: For simulating spiking neural networks.
  - `hypernetx`: For working with hypergraphs.
  - `configparser`: For configuration file parsing.
  - `matplotlib`: For plotting and visualization.

### Installation

To set up the environment, use the following commands:

```bash
pip install -r requirements.txt
```
for installing the required libraries to run QuantumCryptoAgent.py just run the install_script.sh file

```bash
sh install_script.sh
```
