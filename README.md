# OCTR Safe RL HVAC Optimization

This repository contains the first prototype of **OCTR Technologies'** formally verified Safe Reinforcement Learning (SRL) architecture for autonomous building energy optimization.

## Objective
To minimize energy utilization intensity while strictly enforcing the thermal comfort parameters defined by **ASHRAE Standard 55**. The system guarantees a Predicted Mean Vote (PMV) metric between **-0.5 and +0.5**, ensuring the Predicted Percentage of Dissatisfied (PPD) remains securely below 10% regardless of transient external conditions.

## Architecture
Traditional intelligent HVAC systems exhibit a critical vulnerability: they often make oscillating or physically detrimental decisions when fed noisy, subjective, or contradictory feedback from facility managers and tenant comfort applications. 

To resolve this, our environmental control problem is formulated as a **Constrained Markov Decision Process (CMDP)** secured by **Neural Barrier Certificates** and **Control Barrier Functions (CBFs)**.

Instead of allowing the neural network to aggressively chase transient tenant complaints, our safety protocol mathematically bounds the Lipschitz constants of the building's thermal dynamics.

- **Deterministic Fall-safe:** In the event of sensor discrepancies, adversarial inputs, or conflicting feedback loops, the algorithm is constrained from taking rash exploratory actions. It seamlessly engages a deterministic fall-safe, reverting to a proven, physics-informed baseline.
- **NIST AI RMF Compliance:** This architecture adheres strictly to the NIST AI Risk Management Framework by enforcing formal safety verification prior to actuation.

## Repository Structure
- `src/env/`: Contains the Custom Gymnasium CMDP environment for building thermal dynamics.
- `src/cbf/`: Contains the Control Barrier Function safety filters.
- `src/agent/`: Safe RL agents (e.g., PPO + CBF).
- `papers/`: Deep research context and literature review on Safe RL and CBFs in HVAC.

## OmniRush Integration
This architecture is designed to be processed using **OmniRush** for massive multi-zone building simulations. The 5M daily token allocation will allow us to concurrently execute millions of state-transition trajectories, evaluate edge-case boundary conditions, and train our barrier functions in complex simulated environments prior to edge deployment.
