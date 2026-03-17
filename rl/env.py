"""
Quantum_RL/rl/env.py
基于 Gymnasium 的强化学习环境适配层
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy


from utils.Constants import SIMULATION_CONFIG
from core.Network import QuantumNetworkManager
from routing.KSP import QuantumRouter
from core.Request import RequestStatus
from utils import RequestGenerator
from utils.Constants import TOPO_CONFIG, KSP_CONFIG, REQUEST_CONFIG
from utils.rl_config import RLConfig

class QuantumRLEnv(gym.Env):
    """
    量子网络资源调度的强化学习环境 (序列决策模式)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, network: QuantumNetworkManager, router: QuantumRouter, generator: RequestGenerator,
                 max_slots = RLConfig.MAX_SLOTS_TRAIN, max_steps = RLConfig.MAX_STEPS_PER_EPISODE):

        # 1. 直接接管外部传入的物理组件
        self._latest_snapshot = None
        self.network = network
        self.router = router
        self.generator = generator

        # 2. 仿真状态变量直接属于 Env
        self.max_slots = max_slots    #回合的最长时隙
        self.max_wait = SIMULATION_CONFIG.get('MAX_WAIT', 50)  # 超时阈值
        self.lam = SIMULATION_CONFIG.get("LAM",5)
        self.current_slot = -1
        self.total_requests = 0
        self.total_allocated_bits = 0
        self.total_consumed_pairs = 0
        self.W = RLConfig.W          # 前瞻窗口大小：当前请求 + 预见的未来 W-1 个请求
        self.pending_requests = []
        self.completed_requests = []
        self.blocked_requests = []
        # 预加载模式控制开关和流量池
        self.use_preloaded = False
        self.preloaded_pool = {}

        # 提取常量配置
        self.num_nodes = TOPO_CONFIG.get('NUM_NODES',50)
        self.K = KSP_CONFIG.get('K',5)
        self.max_hops = 6  # 假设路径最大跳数，用于 Padding
        self.max_pairs = REQUEST_CONFIG.get('MAX_NUM_PAIRS',5)

        # ==========================================
        # 1. 定义动作空间 (Action Space)
        # ==========================================
        # 长度为 K 的离散数组，每个元素取值 0~self.max_pairs（示例：[2, 0, 1, 0, 0]）
        self.action_space = spaces.MultiDiscrete([self.max_pairs + 1] * self.K)

        # ==========================================
        # 2. 定义观察空间 (Observation Space)
        # ==========================================
        self.observation_space = spaces.Dict({
            # 全局节点资源矩阵 [N, 2]: [剩余速率 Re, 剩余存储 M]   归一化
            "network_resources": spaces.Box(low=0, high=1.0, shape=(self.num_nodes, 2), dtype=np.float32),

            # 前W条请求的 K 条路径特征 [W, K, max_hops'6', 4]: [预期损耗, 剩余速率, 预期存储, 剩余存储]   归一化
            "request_paths": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.W, self.K, self.max_hops, 4),
                dtype=np.float32
            ),

            # 前W条请求的剩余需求量 [W]   归一化
            "request_demand": spaces.Box(
                low=0, high=1.0,
                shape=(self.W,),
                dtype=np.float32
            ),

            # 全局先验上下文 [2]: [当前时隙所有请求的总需求量, 当前请求已等待的时隙数]   归一化
            "global_context": spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        })

        # 环境运行状态管理
        self.current_pending_queue = []
        self.current_request = None
        self.slot_total_demand = 0
        # 强制步数限制
        self.current_step_count = 0
        self.max_steps_per_episode = max_steps

    def reset(self, seed=None, options=None):
        """
        重置环境，开始一个新的 Episode（即一次完整的仿真运行）
        """
        super().reset(seed=seed)

        # 1. 重新初始化环境自身的仿真状态 (彻底剥离 self.sim)
        self.current_slot = -1
        self.total_allocated_bits = 0
        self.total_consumed_pairs = 0
        self.total_requests = 0

        # 清空各类请求队列
        self.pending_requests.clear()
        self.completed_requests.clear()
        self.blocked_requests.clear()

        # 2. 重置底层网络（物理资源恢复满血）
        self.network.reset_resources()
        # 3. 清空环境内部运行状态 (防呆设计，确保不遗留上个 Episode 的数据)
        self.current_pending_queue.clear()
        self.current_request = None
        self.slot_total_demand = 0
        # 4. 重置步数计数器
        self.current_step_count = 0

        # 5. 步进到有请求到达的第一个时隙，开始新一轮的模拟
        self._advance_to_next_active_slot()

        return self._get_observation(), {}

    def step(self, action):
        """
        智能体执行多径分配 (Multi-path Allocation)
        action: 长度为 K 的数组，例如 [2, 0, 1, 0, 0]
        """
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        self.current_step_count += 1
        req = self.current_request

        # 安全校验（理论上不会触发，除非仿真自然结束还要强行step）
        if req is None:
            return self._get_observation(), 0.0, True, False, info

        initial_remaining = req.remaining_pairs
        total_allocated_this_step = 0
        failed_attempts = 0
        attempted_any = False

        # ==========================================
        # 1. 启发式动作截断与多径执行
        # ==========================================
        for k_idx in range(self.K):
            alloc_amount = action[k_idx]

            # 如果智能体不想在这条路分配，或者需求已经被满足了，直接跳过
            if alloc_amount == 0 or req.remaining_pairs <= 0:
                continue

            # 如果这条路径在拓扑中根本不存在（无效路径），直接跳过
            if k_idx not in req.paths:
                continue

            attempted_any = True

            # 【启发式截断】：绝不允许多分，最多分到 remaining_pairs
            actual_alloc = min(alloc_amount, req.remaining_pairs)

            # 调用底层纯净 API 尝试分配
            # (注意: 在 Network.py 的 attempt_allocate_resource_rl 里, 已经实现了 req.remaining_pairs 的扣减)
            is_success, physical_cost = self.network.attempt_allocate_resource_rl(
                req, path_index=k_idx, attempt_amount=actual_alloc
            )

            if is_success:
                total_allocated_this_step += actual_alloc
                self.total_allocated_bits += actual_alloc
                self.total_consumed_pairs += physical_cost
            else:
                # 记录资源溢出导致的失败次数
                failed_attempts += 1
        # ==========================================
        # 2. 状态更新与队列流转
        # ==========================================
        if req.remaining_pairs <= 0:
            self.completed_requests.append(req)
        else:
            # 没分完（可能是一次没分完，也可能是动作被全部驳回），塞回池子等下一轮
            self.pending_requests.append(req)
        # ==========================================
        # 3. 宏观奖励结算 (Reward Shaping - 核心防摆烂设计)
        # ==========================================
        if total_allocated_this_step > 0:
            # [正向激励] 基础分配奖励: 每成功分配1个纠缠对，给 10 分
            reward += total_allocated_this_step * 10.0

            # [完结大奖] 如果这个请求在本轮被彻底满足了 (剩余需求归零)
            if req.remaining_pairs <= 0:
                # 完结奖励是基础奖励的4倍
                reward += 200 + req.num_pairs * 10.0
        # [越界惩罚]越界不仅代表着要扣分，还代表正向奖励一分不得
        if failed_attempts > 0:
            reward -= failed_attempts * 0.1

        # 防摆烂惩罚：打破全写0拿0分的局部最优解，只要尝试了就不会触发
        if not attempted_any and initial_remaining > 0:
            reward -= 2.0
        reward = reward / 10
        # ==========================================
        # 4. 步数截断与时隙推进
        # ==========================================
        if self.current_step_count >= self.max_steps_per_episode:
            truncated = True  # 告诉 PPO：“强行拔电源了，开始算分吧！”

        # 弹出下一个聚光灯下的请求
        if len(self.current_pending_queue) > 0:
            self.current_request = self.current_pending_queue.pop(0)
        else:
            # 队列空了，说明本时隙的所有请求都过了一遍，推进时隙！
            terminated = self._advance_to_next_active_slot()

            # 💡 核心逻辑：检查有没有新拍的快照
            if hasattr(self, '_latest_snapshot') and self._latest_snapshot is not None:
                info['snapshot'] = self._latest_snapshot
                self._latest_snapshot = None

        return self._get_observation(), reward, terminated, truncated, info

    def _advance_to_next_active_slot(self):
        """
        内部核心逻辑：暗中推进时隙，直至队列中出现需要处理的请求，或者到达最大时隙结束仿真。
        此方法负责处理时间的真实流逝、老化、新流量接入，并为 step() 提供排好序的 Request。
        """
        # 物理资源复位 (新时隙开始，所有节点的纠缠生成率和存储器恢复满血)
        self.network.reset_resources()
        # 使用 while 循环：如果某个时隙既没有积压请求，也没有新到达的请求，直接光速跳过，节省大量算力！
        while len(self.current_pending_queue) == 0:
            # 1. 时间指针正式推进
            self.current_slot += 1

            # 2. 包装CSV数据
            if self.use_preloaded and self.current_slot > 0 and self.current_slot % 10 == 0:
                duration = self.current_slot
                sr = len(self.completed_requests) / self.total_requests if self.total_requests > 0 else 0
                tp = self.total_allocated_bits / duration if duration > 0 else 0
                eff = self.total_allocated_bits / self.total_consumed_pairs if self.total_consumed_pairs > 0 else 0
                # 存入内部私有变量
                self._latest_snapshot = [
                    self.current_slot - 1,
                    self.total_requests,
                    len(self.completed_requests),
                    len(self.pending_requests),
                    len(self.blocked_requests),
                    f"{sr:.4f}",
                    f"{tp:.4f}",
                    f"{eff:.6f}"
                ]

            # 3. 终止条件检查
            if self.current_slot >= self.max_slots:
                return True  # 返回 True 告诉 step() 仿真彻底结束 (Terminated)

            # 4. 老化逻辑 (Aging) - 处理上一时隙 step() 没满足而塞回来的积压请求
            alive_reqs = []
            for r in self.pending_requests:
                r.waiting_slots += 1
                if r.waiting_slots > self.max_wait:
                    r.status = RequestStatus.BLOCKED_EXPIRED
                    self.blocked_requests.append(r)
                else:
                    alive_reqs.append(r)
            self.pending_requests = alive_reqs

            # 5. 新流量接入 (Arrivals)
            if self.use_preloaded:
                # 【评估测试模式】：从 RLScheduler 提前灌入的流量池中拿取考卷
                original_requests = self.preloaded_pool.get(self.current_slot, [])
                if original_requests:
                    new_requests = copy.deepcopy(original_requests)  # 必须深拷贝，防止污染原始考卷
                else:
                    new_requests = []
            else:
                # 【训练模式】：用生成器随机生成符合泊松分布的无尽盲盒流量
                new_requests = self.generator.generate_requests_poisson(self.lam, start_id=self.total_requests)

            self.total_requests += len(new_requests)

            # 为新到达的请求计算路由 (KSP)
            for r in new_requests:
                self.router.k_shortest_paths(r)
                # 只有成功找到路由的，才有资格进入候选队列
                if r.status == RequestStatus.ROUTED:
                    self.pending_requests.append(r)
                else:
                    self.blocked_requests.append(r)

            # 6. 💡 启发式先验排序 (Expert Prior) 💡
            # 按照每个请求“最好的一条路”的 UEC 损耗进行从小到大排序。
            self.pending_requests.sort(
                key=lambda x: min(x.candidate_costs.values()) if x.candidate_costs else float('inf')
            )

            # 7. 装填本时隙的决策工作台
            self.current_pending_queue = copy.copy(self.pending_requests)
            # 清空暂存区，因为等下由 step() 函数负责把没分完的请求重新塞回 pending_requests
            self.pending_requests.clear()

            # 8. 计算全局上下文：本时隙的宏观总压力 (传给观察空间，让智能体知道当下拥塞程度)
            self.slot_total_demand = sum(r.remaining_pairs for r in self.current_pending_queue)

        # current_pending_queue 决策工作台
        # 弹出排在第一名的请求，将聚光灯打在它身上，供下一步 step() 函数进行多径裁决
        self.current_request = self.current_pending_queue.pop(0)

        return False  # 返回 False 表示仿真还未结束

    def _get_observation(self):
        """
        提取当前的观测状态 (State) - 终极归一化 + 显式物理特征版本
        """
        # 如果仿真结束，返回空状态
        if self.current_request is None:
            return {
                "network_resources": np.zeros((self.num_nodes, 2), dtype=np.float32),
                "request_paths": np.full((self.W, self.K, self.max_hops, 4), -1.0, dtype=np.float32),
                "request_demand": np.zeros((self.W,), dtype=np.float32),
                "global_context": np.zeros(2, dtype=np.float32)
            }

        curr_req = self.current_request

        # ==========================================
        # 归一化常数定义 (根据物理环境配置)
        # ==========================================
        MAX_RATE_GLOBAL = 500.0
        MAX_MEM_GLOBAL = 100.0
        MAX_WAIT_SLOTS = 50.0
        MAX_SLOT_DEMAND = 600.0   #200*3，200是pending池，3是平均需求量

        # 1. 全局资源矩阵 [归一化到 0.0 ~ 1.0]
        res_matrix = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for i in range(self.num_nodes):
            node = self.network.get_node_object(i)
            # 除以全局最大物理上限，保留骨干与边缘的绝对差异
            res_matrix[i, 0] = (node.max_rate - node.current_rate) / MAX_RATE_GLOBAL
            res_matrix[i, 1] = (node.max_memory - node.current_memory) / MAX_MEM_GLOBAL

        # [W, K, max_hops, 4] 路径特征矩阵 (无效节点保持 -1.0)
        path_matrix = np.full((self.W, self.K, self.max_hops, 4), -1.0, dtype=np.float32)
        # [W,] 需求量矩阵 (无效位保持 0.0)
        demand_matrix = np.zeros((self.W,), dtype=np.float32)
        lookahead_requests = [curr_req]  # 0号位永远是当前正在处理的请求
        lookahead_requests.extend(self.current_pending_queue[:self.W - 1])

        # 2. 路径特征矩阵 [W, K, max_hops, 4]: [预期损耗, 剩余速率, 预期存储, 剩余存储]  无效节点保持 -1.0 掩码
        for w_idx, loop_req in enumerate(lookahead_requests):
            # 3. 需求量 [归一化到 0.0 ~ 1.0]
            demand_matrix[w_idx] = loop_req.remaining_pairs / max(1.0, float(self.max_pairs))
            for k_idx in range(self.K):
                if k_idx in loop_req.paths and k_idx in loop_req.UEC_eff:
                    path_nodes = loop_req.paths[k_idx]
                    node_losses = loop_req.UEC_eff[k_idx]

                    if len(path_nodes) > self.max_hops:
                        continue
                    if any(node_losses.get(node_id, 0.0) == float('inf') for node_id in path_nodes):
                        break
                    for hop_idx, node_id in enumerate(path_nodes):
                        if hop_idx >= self.max_hops: break    # 底层的UEC_eff是进行升序排序的
                        node = self.network.get_node_object(node_id)
                        # 取出当前节点和上一跳节点的损耗
                        loss_curr = node_losses.get(node_id, 0.0)
                        loss_prev = node_losses.get(path_nodes[hop_idx - 1], 0.0) if hop_idx > 0 else 0.0

                        estimated_re_needed = loss_curr * loop_req.remaining_pairs
                        estimated_mem_needed = (loss_curr + loss_prev) * loop_req.remaining_pairs

                        re_pressure = estimated_re_needed / max(1.0, node.current_rate)
                        mem_pressure = estimated_mem_needed / max(1.0, node.current_memory)

                        # 特征归一化
                        norm_loss = min(re_pressure, 1.0)
                        norm_rate = (node.max_rate - node.current_rate) / MAX_RATE_GLOBAL
                        norm_pressure = min(mem_pressure, 1.0)
                        norm_mem = (node.max_memory - node.current_memory) / MAX_MEM_GLOBAL

                        # 加入 w_idx 维度，将数据精准填入 4 维张量！
                        path_matrix[w_idx, k_idx, hop_idx, 0] = norm_loss
                        path_matrix[w_idx, k_idx, hop_idx, 1] = norm_rate
                        path_matrix[w_idx, k_idx, hop_idx, 2] = norm_pressure
                        path_matrix[w_idx, k_idx, hop_idx, 3] = norm_mem

        # 4. 全局上下文 [归一化到 0.0 ~ 1.0]
        norm_total_demand = min(self.slot_total_demand / MAX_SLOT_DEMAND, 1.0)
        norm_wait = min(curr_req.waiting_slots / MAX_WAIT_SLOTS, 1.0)
        global_ctx = np.array([norm_total_demand, norm_wait], dtype=np.float32)

        return {
            "network_resources": res_matrix,
            "request_paths": path_matrix,
            "request_demand": demand_matrix,
            "global_context": global_ctx
        }