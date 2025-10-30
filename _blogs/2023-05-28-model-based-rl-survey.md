---
title: A Survey of Model-Based Reinforcement Learning
date: 2023-05-28
layout: blog
permalink: /blogs/2023-05/model-based-rl-survey/
tags:
  - RL
  - Survey
  - Model-Based
  - English
intro: This survey reviews advances in model-based reinforcement learning (MBRL), summarizing model learning and policy optimization frameworks.
---

## Abstract

In recent years, development in Reinforcement Learning (RL) contributes to super-human performances in games, such as Go, Chess, and StarCraft, as well as in daily contexts, including conversation and robot control. However, RL learns through trial-and-error, which potentially requires a huge amount of data. A new line of research called model-based RL (MBRL) can largely alleviate the data requirement both empirically and theoretically. This survey reviews the recent advances in MBRL mainly from two perspectives, i.e., how to obtain a model and how to obtain a policy, whereas the latter topic can be further divided into learning-free and learning-based sections. The survey ends with a brief discussion of future directions.

## Introduction

### Reinforcement Learning

In recent years, development in Reinforcement Learning (RL) contributes to super-human performances in games, such as Go[^silver2016mastering], Chess, and StarCraft, as well as in daily contexts, including conversation and robot control. Especially, large language models combined with RL with Human Feedback (RLHF) show great potential in aligning with human intentions and are in some ways shaping the world we live in. 

Despite its great influence, RL itself seems straightforward. The goal of RL is to find a policy that maximizes the sum of future rewards. Formally, a Markovian Decision Process (MDP) is denoted as $\langle S, A, P, R, \gamma \rangle$, where $S$ denotes the state, $A$ the action, $P$ the transition probability, $R$ the reward, and $\gamma$ the discount factor. At each state $s_t$, the agent takes an action $a_t$ and observes the next state $s_{t+1}=f(s_t, a_t)$ as well as a reward $r_t$. The goal is to maximize future rewards: $G_t = \sum_i \gamma^i r_{t+i}$. 

### Model-Based Reinforcement Learning

However, common model-free RL algorithms learn purely from trial-and-error, which requires large amounts of data and direct interaction with the world (low sample efficiency).

To address the sample efficiency problem, we turn to Model-Based RL (MBRL), whose core framework is:

```

#### Common Framework of MBRL

```text
while not converged:
    1. Collect data D under the current policy Π
    2. Learn a model M with the collected data
    3. Improve the policy Π using the model M
```

Most MBRL algorithms divide the problem into two separate processes: learning a world model that represents transition (and sometimes reward) dynamics, and learning a policy (either parameterized or via planning/search).

## Paper Organization

We begin with model learning: methods that range from simple one-step transitions to structures like transformers. Next, we survey model usage—learning-free policies (random shooting, Monte Carlo tree search, LQR), then learning-based policies (direct gradient, model-free RL, imitation learning). The survey ends with a discussion of future directions.

## Model Learning

The core of MBRL is model learning. An ideal model accurately predicts the future and handles high-dimensional, multi-modal, or complex observations.

### Transition Model

The most straightforward method is to directly model the one-step transition $s_t = f(s_t, a_t)$, e.g., via supervised learning to minimize mean-squared error between predictions and real states. To avoid distribution mismatch, roll out the current policy to add on-policy data. Model compounding errors are usually handled via Model Predictive Control (MPC): execute only the first action in a predicted sequence and replan every step. This gives rise to the following algorithm skeleton:

```text
1. Run policy Π(a_t|s_t) to collect dataset D = {(s, a, s')}
2. while not converged:
    a. Train the dynamics model f(s, a) on dataset D (minimize loss)
    b. Rollout policy via f(s, a) to plan actions
    c. Execute first planned action, observe resulting s'
    d. Add (s, a, s') to D
```

Early algorithms using such procedures achieve strong results (e.g., Mb-Mf). Improvements include ensemble models for uncertainty, VAE-structured latent models for high-dimensional data, and recurrent/transformer-based architectures.

### Model Uncertainty

Using a single transition model often leads to overfitting and model bias. MBRL algorithms commonly address this by:
- Training an ensemble of models and evaluating their consistency.
- Attenuating policy updates or sample weights based on epistemic (model) uncertainty.

Recent works show that using an ensemble has benefits mainly through regularization; single models with strongly regularized value functions can sometimes match or outperform ensembles.

### Variational Auto-Encoder and Latent Models

To cope with high-dimensional or partially observed settings, variational auto-encoder (VAE) models extract latent features from observations, making planning and prediction more tractable. Classic workflows include multiple losses: latent dynamics, reconstruction, and reward. 

Recent approaches use Recurrent State Space Models (RSSM), where state includes a hidden variable and a posterior variable, allowing the model to encode time-structure and image content jointly. Dreamer-family algorithms discretize or otherwise structure these latent spaces for additional performance gains.

![Structure of the Recurrent State Space Model (RSSM)]({{ '/assets/images/2023-05-28/worldmodel.png' | relative_url }})

## Planning and Searching

With an accurate model, planning or searching algorithms enable super-human results with efficient training. This section presents three classic learning-free algorithms: random shooting, Monte Carlo tree search (MCTS), and linear quadratic regulator (LQR).

### Random Shooting

Random shooting is simple but effective for low-dimensional, short-horizon tasks. The agent randomly generates K candidate action sequences, simulates rewards for each, and selects the most promising one. Real-world use often improves performance with the Cross Entropy Method (CEM), which repeatedly samples and fits a distribution to elite candidates.

```text
while in action optimization:
    1. Sample K candidate action sequences from distribution p(A)
    2. Evaluate predicted rewards for each
    3. Select top M elite sequences
    4. Update p(A) to fit elites
    5. Repeat until done
```

### Monte Carlo Tree Search (MCTS)

MCTS combines planning and search, and is crucial in AlphaGo, AlphaZero, and MuZero. It builds a tree that simulates many possible futures, balancing exploration and exploitation at each node. Policy networks, value networks, and learned models can all be combined via MCTS.

![Steps of Monte Carlo Tree Search]({{ '/assets/images/2023-05-28/MCTS-steps.svg.png' | relative_url }})

### Linear Quadratic Regulator (LQR)

LQR comes from control theory. The agent minimizes a quadratic cost in state (`x_t`) and action (`u_t`), assuming linear-Gaussian dynamics:

- Backward recursion computes optimal gains with system matrices.
- Forward recursion rolls out the best actions using these gains.

## Policy Learning

Model-based RL can also use parameterized policies learned by:
- Direct gradient (backpropagating rewards through the learned model). Gradient-based fine-tuning resembles the PILCO family and Dreamer.
- Model-free RL within a learned model environment—e.g., DYNA, q-learning, actor-critic, and actor-critic with ensembles.
- Imitation learning and distillation—using expert data or controllers as initial policy, or using ensembles for robust initialization.

Methods such as Dreamer, MBPO, SLBO, and others combine these approaches with model-based rollouts, direct gradient, or other hybridizations.

## Conclusion

Sample efficiency in RL remains a major research target, and MBRL is an effective approach. MBRL research focuses on model learning (dynamics, uncertainty, latent space, transformers) and policy application (planning, searching, policy gradient, actor-critic, imitation). As hardware and model architectures advance, so too does the practical power of MBRL in control and AI planning.

## References
- [1] Sutton, R. S., and Barto, A. G. (2018). Reinforcement Learning: An Introduction. 2nd Edition.
- [2] Nagabandi, A., Kahn, G., Fearing, R. S., and Levine, S. (2018). Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning. ICRA.
- [3] Hafner, D., Lillicrap, T., Norouzi, M., and Ba, J. (2019). Dream to Control: Learning Behaviors by Latent Imagination. arXiv:1912.01603.
- [4] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science.
- [5] Janner, M., Fu, J., Zhang, M., and Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. NeurIPS.
