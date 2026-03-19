# 基于深度强化学习与 UEC 指标的量子网络路由优化
# Quantum Network Routing via Deep Reinforcement Learning and UEC Metric

🌍 *Read this in other languages: [中文](#中文) | [English](#english)*

---

<h2 id="中文">🇨🇳 中文说明 (Chinese Version)</h2>

本项目旨在解决连续变量量子密钥分发 (CV-QKD) 网络中，由于节点存储器容量受限而导致的并发通信请求资源竞争问题。项目包含两阶段的核心工作：
首先提出了一种全新的量子路由度量模型 **UEC (Unit Entanglement Consumption)** 并验证了其在传统启发式算法中的优越性；
随后，为了弥补传统算法缺乏“大局观”的缺陷，构建了基于 UEC 先验引导的深度强化学习 (DRL) 路由调度框架。

### 📌 第一阶段：UEC 度量模型与传统贪婪调度

在项目的基础构建阶段，我们不仅完成了量子网络底层架构（节点、链路、保真度衰减等）的搭建，更重要的是提出了一种全新的路由度量模型——**非纠缠成本 (UEC)**。
* **物理意义映射**：UEC 能够将端到端的量子保真度约束，精准映射为节点级的物理资源消耗量。
* **资源池调度**：基于 UEC，网络的路由调度可以转化为基于节点资源池的分配问题，从而在很大程度上缓解了存储器容量瓶颈导致的拥塞。

为了评估该指标，我们构建了 `GreedyScheduler.py`。该调度器利用 `PathCandidate.py` 进行极其严苛的贪婪寻路与调度。
实验结果表明，在传统贪婪算法框架下，**UEC 相比最短跳数（MinHops）、竞争率（CR）等传统度量表现出显著性能优势**，可取得传统启发式方法中的最优效果。

### 🧠 第二阶段：为何引入强化学习 (RL)？

尽管基于 UEC 的贪婪算法表现优异，但在实际的高动态量子网络中仍存在致命局限：
1. **缺乏“审时度势”的大局观**：贪婪算法极度短视，只能针对当前单一请求做出局部最优决策，无法为未来即将到来的优质请求预留核心骨干资源，容易导致阻塞与长尾请求饥饿。
2. **缺乏动态鲁棒性**：量子物理链路存在极高的随机噪声，静态的贪婪数学公式难以应对突发的网络拥塞坍缩。
3. **高维特征利用率不足**：UEC 指标及网络状态本身是一个极其复杂的高维特征向量（包含剩余容量、预期损耗、全局需求等），传统数学排序无法完全挖掘其中的非线性关联，而这正是**深度神经网络 (DNN)** 最擅长的领域。

因此，本项目的第二阶段核心是：**构建一个将 UEC 机制深度融入“状态-动作空间 (State-Action Space)”的强化学习 (PPO) 路由统筹大脑。**

### 💡 核心贡献与创新点

1. **UEC 驱动的动作空间重塑与掩码 (Action Space Shaping & Masking)**：
   我们并非仅仅将 UEC 作为外部启发式规则，而是将其作为构建 RL 动作空间的物理基石。利用 UEC 对 K-Shortest Path (KSP) 进行严格的物理级预过滤，这不仅是对无效动作的精准掩码，更从根本上重塑了智能体可执行动作的边界，极大加速了模型的收敛。
2. **基于 UEC 特征融合的高维观察空间 (Observation Space)**：
   定制专属的 `[512, 256, 128]` 大容量特征提取器，完美融合了包含网络拓扑资源、**基于 UEC 评估的路径特征**和全局上下文的 1300+ 维状态张量，赋予了智能体极强的底层物理直觉。
3. **前瞻视窗 ($W=10$)**：
   引入滑动窗口状态表示机制。RL 智能体不仅能感知当前请求，还能“看穿”未来 9 个排队中的请求画像，实现跨时隙的全局统筹与核心资源预留。
4. **两阶段混合残差调度架构 (Two-Stage RL-Augmented Heuristic Scheduling)**：
    创新性地提出了一种“宏观统筹+微观扫尾”的隐式竞合（Coopetition）混合架构。RL 智能体充当统筹者，专注于跨时隙的长尾拥塞博弈与战略性骨干资源分配；
    而在每个时隙末尾，利用 UEC 贪婪算法作为“残差安全网（Residual Safety Net）”榨干物理碎片。
    通过精心设计的“惩罚退还”机制诱导智能体学会任务委托，在为系统提供极高物理保底（Lower-bound）的同时，有效突破了极限吞吐量上限。

### 📂 项目结构

* `core/`: 量子网络底层物理组件（节点、链路、网络图）。
* `rl/`: 强化学习交互环境 (`env.py`)、自定义特征提取器及超参数配置。
* `routing/`: K-最短路 (KSP) 及各种路由度量指标的实现（重点包含 **`UEC.py`**, `MinHops.py`, `CR.py`）。
* `scheduler/`: 调度器模块，包含传统贪婪调度 (`GreedyScheduler.py`) 与 RL 调度 (`RLScheduler.py`)。
* `experiments/`: 训练日志、评估结果 (.csv) 以及保存的模型权重文件。

### 🚀 快速开始与评估说明

**1. 环境依赖配置**
本项目基于 **Python 3.10** 开发与深度测试。请在您的 Python 虚拟环境中执行以下命令安装核心依赖包：
`pip install torch numpy gymnasium stable-baselines3 tensorboard pandas networkx matplotlib`

**2. 训练强化学习智能体**
要从头开始训练深度融合了 UEC 机制的 PPO 统筹智能体，请在终端运行：
`python run_rl_train.py`

**3. 模型评估与基线对比 (`evaluate_model.py` 必读指南)**
`evaluate_model.py` 是本项目的核心测试脚本，内部包含两套完全独立的评估逻辑，分别用于对比“传统算法”与“强化学习算法”：

* **生成传统启发式算法的基线数据 (Baselines)**：
  代码中的 `run_experiment()` 函数调用了 `GreedyScheduler`。它无需加载任何神经网络模型，会直接在工程根目录下生成传统数学规则的运行结果 CSV 文件。若需运行传统基线算法，请确保取消代码中的注释。

* **评估已训练的强化学习模型 (RL Agent)**：
  代码底部的 `scheduler.evaluate(...)` 方法专门用于测试强化学习智能体的动态调度能力。它会加载指定路径下的 `.zip` 模型参数，并在 `experiments/results/` 目录下输出评估报告。
  `scheduler.evaluate(model_path=MODEL_PATH, test_pool=request_dataset)`
* **两阶段混合架构 (Hybrid Architecture Evaluation)**：
  新增训练包含贪婪扫尾的混合智能体 (Hybrid Agent)：python run_hybrid_train.py
  运行 evaluate_hybrid_model.py，该脚本将调用核心模块 HybridScheduler.py。
  在此模式下，预训练的 RL 智能体将与贪婪算法协同作战（RL 负责高ROI请求分配，Greedy 负责时隙末尾的兜底捡漏），在 experiments/results/ 目录下输出融合架构的极限界限评估报告。

---

<h2 id="english">🇺🇸 English Version</h2>

This project aims to solve the resource competition problem of concurrent communication requests caused by limited node memory capacity in continuous-variable quantum key distribution (CV-QKD) networks. The project consists of two core phases: First, proposing a novel routing metric model **UEC (Unentanglement Cost)** and validating its superiority in traditional heuristic algorithms; Second, constructing a Deep Reinforcement Learning (DRL) routing scheduling framework guided by UEC priors to overcome the myopic nature of traditional algorithms.

### 📌 Phase 1: UEC Metric and Traditional Greedy Scheduling

In the foundational phase, we constructed the underlying architecture of the quantum network (nodes, links, fidelity attenuation) and proposed a novel routing metric: **Unentanglement Cost (UEC)**.
* **Physical Mapping**: UEC accurately maps end-to-end quantum fidelity constraints to node-level physical resource consumption.
* **Resource Pool Scheduling**: Based on UEC, network routing is transformed into a node resource pool allocation problem, significantly alleviating congestion caused by memory bottlenecks.

We built `GreedyScheduler.py` to evaluate this metric. Using `PathCandidate.py` for strict greedy pathfinding, experimental results show that under the greedy algorithm framework, **UEC significantly outperforms traditional metrics (such as MinHops and CR)**, achieving the optimal solution in the traditional algorithmic sense.

### 🧠 Phase 2: Why Introduce Reinforcement Learning (RL)?

Despite the excellent performance of the UEC-based greedy algorithm, fatal limitations remain in highly dynamic quantum networks:
1. **Lack of Global Foresight**: Greedy algorithms are extremely myopic, making local optimal decisions for a single request without reserving backbone resources for future high-value requests, often leading to large-scale blockages.
2. **Lack of Dynamic Robustness**: Quantum physical links have high random noise, and static mathematical formulas struggle to cope with sudden network congestion collapses.
3. **Underutilization of High-Dimensional Features**: The UEC metric and network state form a complex high-dimensional feature vector. Traditional mathematical sorting cannot fully mine the nonlinear correlations within it, which is exactly the domain where **Deep Neural Networks (DNN)** excel.

Therefore, the second core phase of this project is: **Constructing a PPO-based RL global routing brain that deeply integrates the UEC mechanism into its State-Action Space.**

### 💡 Core Contributions & Innovations

1. **UEC-Driven Action Space Shaping & Masking**:
   Rather than treating UEC merely as an external heuristic, we establish it as the physical cornerstone of the RL action space. By utilizing UEC to strictly pre-filter K-Shortest Paths (KSP), we achieve precise invalid-action masking and fundamentally reshape the bounds of valid actions, drastically accelerating model convergence.
2. **High-Dimensional Observation Space with UEC Feature Fusion**:
   A customized `[512, 256, 128]` high-capacity feature extractor perfectly fuses the 1300+ dimensional state tensor—comprising topological resources, **UEC-evaluated path features**, and global context—granting the agent profound underlying physical intuition.
3. **Lookahead Window ($W=10$)**:
   Introduces a sliding window state representation. The RL agent perceives not only the current request but also the profiles of the next 9 pending requests, enabling cross-slot global planning and core resource reservation.

### 📂 Repository Structure

* `core/`: Quantum network physical components.
* `rl/`: RL interaction environment (`env.py`), custom feature extractor, and hyperparameter config.
* `routing/`: KSP algorithm and routing metrics implementations (highlighting **`UEC.py`**).
* `scheduler/`: Schedulers including Greedy (`GreedyScheduler.py`) and RL (`RLScheduler.py`).
* `experiments/`: Training logs, evaluated `.csv` results, and saved model weights.

### 🚀 Quick Start & Evaluation Guide

**1. Environment Requirements**
Tested on **Python 3.10**. Install core dependencies:
`pip install torch numpy gymnasium stable-baselines3 tensorboard pandas networkx matplotlib`

**2. Train the RL Agent**
`python run_rl_train.py`

**3. Model Evaluation (`evaluate_model.py` Guide)**
The `evaluate_model.py` script contains two distinct evaluation logics:

* **Generate Traditional Greedy Baseline Data**:
  The `run_experiment()` function calls the `GreedyScheduler`. It directly generates CSV results for traditional algorithms in the root directory.
* **Evaluate the Trained RL Model**:
  The `scheduler.evaluate(...)` method at the bottom is specifically for testing the RL agent. It loads the specified `.zip` model and outputs the report.