"""
最小跳数路由度量
"""
import networkx as nx
from typing import List, Dict, Any
from .BaseMetric import RoutingMetric


class MinHopsMetric(RoutingMetric):
    """
    最小跳数度量指标。
    路径成本 = 路径中边的数量 (跳数)
    边权重 = 1 (每条边权重相同)
    """

    @property
    def name(self) -> str:
        return "MinHops"

    def calculate_path_cost(self, graph: nx.Graph, path: List[int]) -> float:
        """
        计算路径的总跳数（边数）

        Args:
            graph: NetworkX 图对象
            path: 节点路径列表

        Returns:
            float: 路径跳数
        """
        if not path or len(path) < 2:
            return float('inf')

        # 跳数 = 节点数 - 1
        return len(path) - 1

    def get_edge_weight(self, u: int, v: int, edge_attr: Dict[str, Any]) -> float:
        """
        每条边的权重都为 1

        Args:
            u: 起始节点
            v: 结束节点
            edge_attr: 边属性

        Returns:
            float: 边权重 (恒为 1)
        """
        return 1.0