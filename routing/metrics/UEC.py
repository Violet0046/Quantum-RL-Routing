"""
UEC (Unit Entanglement Consumption) 路由度量
"""
import math
import networkx as nx
from typing import List, Dict, Any
from .BaseMetric import RoutingMetric
from core.Link import QuantumLink
from utils.Constants import P_SWAP_DEFAULT
from utils.Constants import KSP_CONFIG

class UECMetric(RoutingMetric):
    current_max_rounds = 2

    def __init__(self):
        super().__init__()
        # 默认值设置为保守值，防止未注入时报错
        self._ctx_fidelity_threshold: float = 0.6
        self._ctx_estimated_hops: float = 1.0

    @property
    def name(self) -> str:
        return "UEC"

    def set_context(self, fidelity_threshold: float, estimated_hops: int):
        """接收 Router 传来的信息"""
        self._ctx_fidelity_threshold = fidelity_threshold
        # 至少为1跳
        self._ctx_estimated_hops = max(float(estimated_hops), 1.0)  

    def calculate_path_cost(self, graph: nx.Graph, path: List[int]) -> tuple[float, Dict[int, float]]:
        base_node_losses = self._calculate_path_cost_base(graph, path)
        return self._compute_uec_eff(graph, path, base_node_losses)

    def _calculate_path_cost_base(self, graph: nx.Graph, path: List[int]) -> Dict[int, float]:
        """
        执行N节点链路损耗 UEC_base 量化算法 (逆向递归)

        Args:
            graph: NetworkX 图对象
            path: 节点路径列表

        Returns:
            Dict[int, float]: 节点损耗字典
        """

        p_swap = P_SWAP_DEFAULT  # 纠缠交换成功率

        # 初始化需求 N_req = 1
        n_cur = 1.0

        # 存储每个节点的损耗
        node_losses: Dict[int, float] = {}

        # 从倒数第二个节点开始，逐层向前计算 (Reverse Recursion)
        # 路径索引: 0(src), 1, ..., k-1, k(dst)
        # 迭代范围: k-1 down to 1 (中间节点)
        for i in range(len(path) - 2, 0, -1):
            u = path[i]
            v = path[i + 1]  # 下游节点

            # 获取链路衰减因子
            edge_data = graph[u][v]
            link: QuantumLink= edge_data.get('object')

            p_link = link.attenuation

            # 中继节点损耗公式: L_i = N_cur / (P_link * P_swap)
            l_i = n_cur / (p_link * p_swap)
            node_losses[u] = l_i  # 记录该节点的损耗

            # 更新上游需求: N_cur = N_cur / P_swap
            n_cur = n_cur / p_swap

        # 源节点损耗 L_0 只需要满足第一跳的链路成功率，不需要进行交换
        edge_data = graph[path[0]][path[1]]
        link: QuantumLink = edge_data.get('object')

        p_link = link.attenuation
        l_0 = n_cur / p_link
        node_losses[path[0]] = l_0  # 记录源节点的损耗

        return node_losses

    def _compute_uec_eff(self, graph: nx.Graph, path: List[int], base_node_losses: Dict[int, float]) -> tuple[float, Dict[int, float]]:
        real_hops = len(path) - 1
        f_target = self._ctx_fidelity_threshold ** (1.0 / real_hops)
        eff_node_losses = {}
        total_eff_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            F_phys = graph[u][v]['object'].fidelity
            beta = self._purification_beta(F_phys, f_target)
            if beta == float('inf'):
                return float('inf'), {u: float('inf')}
            l_base = base_node_losses[u]
            l_eff = l_base * beta
            eff_node_losses[u] = l_eff
            total_eff_cost += l_eff
        target_node = path[-1]
        eff_node_losses[target_node] = 0.0
        return total_eff_cost, eff_node_losses

    def get_edge_weight(self, u: int, v: int, edge_attr: Dict[str, Any]) -> float:
        """
        为 KSP 搜索提供权重
        使用 -log(P_link) 与 保真度 作为权重

        Args:
            u: 起始节点
            v: 结束节点
            edge_attr: 边属性

        Returns:
            float: 边权重
        """
        link: QuantumLink | None = edge_attr.get('object')
        if not link:
                return float('inf')
        
        p_link = link.attenuation    
        f_phys = link.fidelity
        # 防止 log(0)
        if p_link <= 1e-12:
            return float('inf')
        weight_connectivity = -math.log(p_link)

        target_f_hop = self._ctx_fidelity_threshold ** (1.0 / self._ctx_estimated_hops)

        # 估算纯化因子
        beta = self._purification_beta(f_phys, target_f_hop)
        if beta == float('inf'):
            return float('inf') # 避开保真度过低的链路    
        weight_fidelity = math.log(beta)
        total_weight = weight_connectivity + KSP_CONFIG.get("FIDELITY_WEIGHT", 1.0) * weight_fidelity

        return total_weight

    def _purification_beta(self, F_phys: float, f_target: float) -> float:
        """
        内部辅助方法：基于 BBPSSW 协议迭代计算纯化开销系数 beta
        两个作用：
        1.估算纯化因子, 用于边权计算
        2.计算纯化因子, 用于UEC_base到UEC_eff的转换
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
        max_rounds = self.current_max_rounds     #1就饱和了
        
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

