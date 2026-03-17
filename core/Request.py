"""
量子请求类
定义量子网络中的请求数据结构
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum, auto
from typing import List
from core.Link import QuantumLink
from utils.Constants import P_SWAP_DEFAULT


# 定义请求状态枚举
class RequestStatus(Enum):
    PENDING = auto()  # 刚生成，未处理
    ROUTED = auto()  # 已找到路径，等待分配
    ACCEPTED = auto()  # 分配成功 (资源已锁定)
    BLOCKED_TOPOLOGY = auto()  # KSP失败：物理上不可达（孤岛或图不连通）
    BLOCKED_FIDELITY = auto()  # UEC计算失败：路径存在，但物理保真度太低，纯化代价无穷大
    BLOCKED_EXPIRED = auto()  # 失败：超时导致被截断


@dataclass
class QuantumRequest:
    """
    量子网络请求类

    表示一个量子通信请求，包含源节点、目标节点等信息，
    以及 KSP 算法计算的候选路径和节点损耗信息。
    """

    # 基础业务属性
    request_id: int  # 请求唯一标识
    source: int  # 源节点 ID
    target: int  # 目标节点 ID

    # 请求参数
    num_pairs: int  # 请求的量子比特对数量 N_req
    fidelity_threshold: float  # 保真度阈值 F_th

    status: RequestStatus = RequestStatus.PENDING  # 请求状态
    # [调度状态] 根据时隙和costs计算优先级
    candidate_costs: Dict[int, float] = field(default_factory=dict)  # 非UEC Metric的请求在计算UEC_eff时写入该值
    waiting_slots: int = 0
    # [流式处理] 还需要传送多少对 (用于分帧处理)
    remaining_pairs: int = field(init=False)
    # 所有 Metric 通用接口
    paths: Dict[int, List[int]] = field(default_factory=dict)

    # 有保真度约束的候选路径及节点损耗
    # 结构: {路径索引k: {节点ID: 节点损耗值}}
    UEC_eff: Dict[int, Dict[int, Optional[float]]] = field(default_factory=dict)

    def __post_init__(self):
        self.remaining_pairs = self.num_pairs

    """
        标记请求状态方法：
        1.BLOCKED_TOPOLOGY: 拓扑不可达
        2.BLOCKED_FIDELITY: 保真度不够
    """

    def mark_as_blocked_topology(self):
        self.status = RequestStatus.BLOCKED_TOPOLOGY
        self.paths.clear()
        self.UEC_eff.clear()

    def mark_as_blocked_fidelity(self):
        self.status = RequestStatus.BLOCKED_FIDELITY

    """
        非UEC Metric的请求类用到的方法:
        1.计算UEC_eff
        2.计算资源占用
    """

    def compute_uec_eff(self, graph):
        """
        为非UEC Metric请求计算UEC_eff
        注意: 不需要根据UEC_eff成本进行重排序
        """
        UEC_base = self._calculate_uec_base(graph)
        UEC_eff: Dict[int, Dict[int, Optional[float]]] = {}
        for k, path in self.paths.items():
            hops = len(path) - 1
            f_target = self.fidelity_threshold ** (1.0 / hops)
            eff_node_losses: Dict[int, float] = {}
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                F_phys = graph[u][v]['object'].fidelity
                beta = self._calculate_purification_beta(F_phys, f_target)
                if beta == float('inf'):
                    eff_node_losses[u] = float('inf')
                    break
                l_base = UEC_base[k][u]
                l_eff = l_base * beta
                eff_node_losses[u] = l_eff
            target_node = path[-1]
            eff_node_losses[target_node] = 0.0
            UEC_eff[k] = eff_node_losses
        self.UEC_eff = UEC_eff

    def _calculate_uec_base(self, graph) -> Dict[int, Dict[int, Optional[float]]]:
        """
        内部辅助方法: 计算UEC_base
        """
        p_swap = P_SWAP_DEFAULT  # 纠缠交换成功率
        UEC_base: Dict[int, Dict[int, Optional[float]]] = {}
        for k, path in self.paths.items():
            n_cur = 1.0
            # 存储每个节点的损耗
            base_node_losses: Dict[int, float] = {}
            for i in range(len(path) - 2, 0, -1):
                u = path[i]
                v = path[i + 1]  # 下游节点

                # 获取链路衰减因子
                edge_data = graph[u][v]
                link: QuantumLink = edge_data.get('object')

                p_link = link.attenuation
                # 中继节点损耗公式: L_i = N_cur / (P_link * P_swap)
                l_i = n_cur / (p_link * p_swap)
                base_node_losses[u] = l_i  # 记录该节点的损耗
                # 更新上游需求: N_cur = N_cur / P_swap
                n_cur = n_cur / p_swap

            # 源节点损耗 L_0 只需要满足第一跳的链路成功率，不需要进行交换
            edge_data = graph[path[0]][path[1]]
            link: QuantumLink = edge_data.get('object')

            p_link = link.attenuation
            l_0 = n_cur / p_link
            base_node_losses[path[0]] = l_0  # 记录源节点的损耗
            UEC_base[k] = base_node_losses
        return UEC_base

    def _calculate_purification_beta(self, F_phys: float, f_target: float) -> float:
        """
        内部辅助方法: 基于 BBPSSW 协议迭代计算纯化开销系数 beta
        """
        if F_phys >= f_target:
            return 1.0
        if F_phys <= 0.5:
            return float('inf')
        # 纯化模拟变量
        f_curr = F_phys
        total_rounds = 0
        cumulative_prob = 1.0  # 累积成功率

        # 设定最大迭代防止死循环 (物理上可能无法达到目标)
        max_rounds = 10  # 这里的迭代次数无所谓，因为是非UEC Metric

        while f_curr < f_target:
            if total_rounds >= max_rounds:
                # 无法达到目标保真度，返回无穷大成本
                return float('inf')

            # BBPSSW 迭代公式
            p_succ = f_curr * F_phys + ((1 - f_curr) * (1 - F_phys))

            # 保真度更新公式
            f_new = (f_curr * F_phys) / p_succ

            f_curr = f_new
            cumulative_prob *= p_succ
            total_rounds += 1
        beta = (total_rounds + 1) / cumulative_prob
        return beta
