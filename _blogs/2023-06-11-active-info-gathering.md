---
title: Active Information Gathering Agent
date: 2023-06-11
layout: blog
tags:
  - RL
  - Active Exploration
  - Model-Based
  - MuJoCo
  - Chinese
intro: 本文系统梳理了主动信息收集智能体（Active Information Gathering Agent）的理论基础及其在RL场景下的算法框架，突出信息增益在policy中的作用及MuJoCo实验结果。
---

## 摘要

强化学习（RL）在智能体适应特定环境和动力学方面长期面临“样本效率”和“泛化能力”难题。本工作受启发于人类主动探索未知环境的能力，提出在智能体策略目标中直接纳入“期望信息增益”，鼓励RL智能体自主探索、主动适应，并在MuJoCo等动态可变环境中验证性能效果。该方法在同等样本下，获得2倍性能提升且样本量不足1/3。

## Introduction

现实机器人控制中的主要挑战在于测试阶段能否快速适应新环境，以及复杂系统动态。人类面对新情境会主动测试/收集信息（如试着加速、刹车），而RL智能体常常缺乏这种“自主探查”的能力。本研究提出激励RL智能体主动交互环境以提升泛化和Sim2Real落地效果。智能体通过最大化策略下的信息增益，适应不同场景、动力学，显著缩小了仿真与现实世界的样本差距。

## 方法：主动信息收集框架

主动信息收集RL智能体框架主要关注两个核心问题：
- 如何度量信息增益？
- 如何利用信息增益调整策略？

### 信息增益的度量

本工作将信息增益形式化为：
$$
I = H(s) - H(s|o)
$$
其中 $s$ 是状态的潜在表示，$o$ 是当前观测，$H$ 是熵。我们通过适应模块（Adaptation Module）逼近信息增益：
$$
I = f(s_t | o_{t-1}, a_{t-1}, ...) - f(s_t | o_t, a_t, o_{t-1}, a_{t-1}, ...)
$$

三种常见设计：
1. **EPI类方法**：轨迹编码器 $f(s_t|o_t,a_t,...)$ 直接在连续轨迹上学习信息增益。
2. **RMA类方法**：轨迹编码器与特权信息编码器协同训练（后者只在训练阶段可见）。
3. **Dreamer类方法（RSSM）**：状态 $s$ 分为循环隐藏态 $h_t$ 和后验 $z_t$，用RNN/变分方法学习潜空间动力学。

三类世界模型结构对比如下：

![三类世界模型结构草图（左：EPI/RMA，右：Dreamer RSSM细节）]({{ '/assets/images/2023-06-11/left.png' | relative_url }})
![RSSM模型推理与生成流程]({{ '/assets/images/2023-06-11/right.png' | relative_url }})

### 主动收集的算法伪代码

```text
Initialize Adaptation Module f(s_t|o_t,a_t,...), RL policy π(a_t|s_t)
while policy not converged:
    Run adjusted policy, collect data
    Train Adaptation Module,抽象潜在状态和重建观测
    用标准RL流程训练policy
    获取包含自适应模块反馈的adjusted policy
```

## 实验与分析

本实验基于 MuJoCo 平台评估主动信息收集框架与PPO基线对比。

### 环境设置

不同环境参数（训练/测试）：

| 参数      | 训练区间           | 测试区间           |
|-----------|-------------------|--------------------|
| Gravity   | [-30, -7]         | [-7, -1]           |
| Friction  | [0.3, 0.9]        | [0.1, 0.3]         |
| Stiffness | [6, 20]           | [2, 6]             |

### 主要实验流程与对比

基线设置：
- **PPO baseline**：多环境训练（无真实参数输入），多环境测试。
- **Privilege PPO**：训练/测试均注入真实环境参数。
- **Normal PPO**：固定单环境训练/测试。

### 曲线与模拟结果

![MuJoCo环境设定与信息收集示意]({{ '/assets/images/2023-06-11/res1.png' | relative_url }})
![信息收集策略的raw training curve]({{ '/assets/images/2023-06-11/res2.png' | relative_url }})

### 主要结论

- 主动信息收集方法在 MuJoCo setting 下最终性能提升 2.0x，所需样本减少 3.2x。
- Privilege PPO 收益 > PPO baseline，但远低于 Normal PPO（说明环境信息极其宝贵）。
- 直接将信息增益作为intrinsic reward虽平滑训练曲线，但最终收敛分数略降（agent探索欲望过强，原任务达成率受影响）。

### OmniDrones 扩展实验

除MuJoCo外，实验还拓展至无人机环境（OmniDrones，含风干扰测试）：

![OmniDrones环境任务示例]({{ '/assets/images/2023-06-11/Screenshot 2023-06-11 at 21.27.18.png' | relative_url }})
![风扰动场景下的信息收集实验曲线]({{ '/assets/images/2023-06-11/W&B Chart 6_6_2023, 1_55_19 PM-2.png' | relative_url }})

主要结论：
- Privilege PPO ≈ Normal PPO > PPO baseline：注入环境信息极大提升表现，风扰下主动适应能力尤甚。
- 提示信息收集机器人（如无人机）实测极易受环境扰动影响，主动探索与自适应方案能帮助表现逼近有标签参数的上限。

## 结论与未来展望

- 主动信息收集智能体显著提升了样本效率与泛化能力。
- 信息增益的设计与利用让RL模型具备更强环境自适应能力，对现实机器人sim2real落地有重要意义。
- 后续可尝试理论证明、全自监督RL、跨任务泛化、复杂物理机器人和现实世界场景实测。

## 主要参考文献
- Hafner et al., 2019. Dream to control: Learning behaviors by latent imagination. arXiv:1912.01603.
- Tobin et al., 2017. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. arXiv:1703.06907.
- Zhou et al., 2019. Environment probing interaction policies. arXiv:1907.11740.
- Kumar et al., 2021. Rapid motor adaptation for legged robots. arXiv:2107.04034.
- Hafner et al., 2023. Mastering Diverse Domains through World Models. arXiv:2301.04104.
- Schulman et al., 2017. Proximal policy optimization algorithms. arXiv:1707.06347.
