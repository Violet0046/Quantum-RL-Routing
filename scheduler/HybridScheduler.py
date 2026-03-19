"""
Quantum_RL/scheduler/HybridScheduler.py
两阶段混合调度器：RL + Greedy
"""
import os
import time
import csv
import random
import numpy as np

from scheduler.RLScheduler import RLScheduler
from rl.env import QuantumRLEnv
from utils.PathCandidate import PathCandidate
from core.Request import RequestStatus
from utils.rl_config import RLConfig
from core.Network import QuantumNetworkManager
from routing.KSP import QuantumRouter
from utils.RequestGenerator import RequestGenerator


class HybridRLEnv(QuantumRLEnv):
    """
    继承自基础 RL 环境，在每个时隙结束时无缝注入贪婪扫尾逻辑
    """

    def step(self, action):
        # 1. 快照备份：在父类流转到下一个请求之前，先捕获当前请求的初始需求量
        req = self.current_request
        initial_remaining = req.remaining_pairs if req is not None else 0

        # 2. 让父类去执行常规的 step
        obs, reward, terminated, truncated, info = super().step(action)

        # 3. 不再添加防摆烂惩罚
        attempted_any = any(alloc > 0 for alloc in action)
        if not attempted_any and initial_remaining > 0:
            # 父类里是 reward -= 2.0，最后 reward = reward / 10，所以实际扣了 0.2
            reward += 0.2

        return obs, reward, terminated, truncated, info

    def _advance_to_next_active_slot(self):
        # 在物理资源(Network)重置、推进到下一个时隙之前！
        # 此时 RL 已经处理完了本时隙所有的请求。我们对残余资源执行贪婪榨干！
        if self.current_slot >= 0:
            self._run_greedy_residual_sweep()

        # 执行完贪婪扫尾后，交回给父类正常推进时隙、老化请求、接入新流量
        return super()._advance_to_next_active_slot()

    def _run_greedy_residual_sweep(self):
        """
        残差贪婪扫尾逻辑：利用 UEC 指标榨干网络最后的物理碎片
        """
        scheduling_pool = []
        alpha = 0.0  # 同样可引入等待时间权重

        # 1. 搜集本时隙 RL 挑剩下的、还没分完的请求
        for req in self.pending_requests:
            for k_idx, cost in req.candidate_costs.items():
                if cost == float('inf'): continue
                score = cost - (alpha * req.waiting_slots)
                cand = PathCandidate(sort_score=score, request=req, path_index=k_idx)
                scheduling_pool.append(cand)

        # 2. 按照 UEC 综合得分从小到大排序
        scheduling_pool.sort()

        # 3. 贪婪“捡漏”分配
        for cand in scheduling_pool:
            req = cand.request
            if req.status == RequestStatus.ACCEPTED or req.remaining_pairs <= 0:
                continue

            # 调用基础 attempt_allocate_resource  尽力而为的分配模式
            allocated_bits, physical_cost = self.network.attempt_allocate_resource(req, cand.path_index)

            if allocated_bits > 0:
                self.total_allocated_bits += allocated_bits
                self.total_consumed_pairs += physical_cost
                if req.remaining_pairs <= 0:
                    req.status = RequestStatus.ACCEPTED

        # 4. 队列清理：把在贪婪阶段完成的请求移入完成队列
        next_slot_pending = []
        for req in self.pending_requests:
            if req.status == RequestStatus.ACCEPTED:
                self.completed_requests.append(req)
            else:
                next_slot_pending.append(req)
        # 留给下一时隙继续处理
        self.pending_requests = next_slot_pending


class HybridScheduler(RLScheduler):
    """
    混合调度器主入口：继承自 RLScheduler。
    只需要重写 build_env 让其返回魔改后的 HybridRLEnv 即可，
    模型加载、测试评估等几百行逻辑代码全部自动复用！
    """

    def build_env(self, max_slots=None, max_steps=None) -> HybridRLEnv:
        # 保持随机数种子一致，确保测试公平性
        random.seed(self.topo_seed)
        np.random.seed(self.topo_seed)

        network = QuantumNetworkManager()
        network.generate_waxman_topology()

        random.seed(None)
        np.random.seed(None)

        router = QuantumRouter(network.graph, self.metric)
        generator = RequestGenerator(network)
        slots = max_slots if max_slots is not None else RLConfig.MAX_SLOTS_TRAIN
        steps = max_steps if max_steps is not None else RLConfig.MAX_STEPS_PER_EPISODE

        # 返回带有贪婪扫尾机制的混合增强环境
        env = HybridRLEnv(network, router, generator, slots, steps)
        return env

    def _prepare_csv_logger(self):
        # 重写日志名，区分"混合"与"纯 RL"
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_filename = f"experiments/results/hybrid_eval_{self.exp_name}_{timestamp}.csv"
        os.makedirs("experiments/results", exist_ok=True)
        headers = ["Slot", "Total_Requests", "Completed", "Pending", "Blocked", "Success_Rate", "Throughput_Avg",
                   "Efficiency_Avg"]
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
        return csv_filename