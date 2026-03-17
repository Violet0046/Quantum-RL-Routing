"""
CR 度量指标
"""
import networkx as nx
from typing import List, Dict, Any
from .BaseMetric import RoutingMetric
from core.Link import QuantumLink

class CRMetric(RoutingMetric):
    """
    基于路径的成功概率，成本 = 1 / (路径成功率)
    """

    @property
    def name(self) -> str:
        return "CR"

    def calculate_path_cost(self, graph: nx.Graph, path: List[int]) -> float:
        """
        计算路径的吞吐量倒数成本。
        """
        if not path or len(path) < 2:
            return float('inf')
        total_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # [修正] 从 QuantumLink 对象获取参数
            edge_data = graph[u][v]
            link: QuantumLink | None = edge_data.get('object', None)
            
            if not link:
                return float('inf')
            
            p_link = link.attenuation # 物理链路成功率
            # 安全检查：防止除以 0
            if p_link <= 1e-12:
                return float('inf')

            # CR 核心逻辑：累加每一跳的倒数
            total_cost += (1.0 / p_link)

        return total_cost

    def get_edge_weight(self, u: int, v: int, edge_attr: Dict[str, Any]) -> float:
        """
        为搜索算法提供边权重
        """
        link: QuantumLink | None = edge_attr.get('object', None)
        if not link:
            return float('inf')
            
        p_link = link.attenuation
        if p_link <= 1e-12:
            return float('inf')

        return 1.0 / p_link