"""
路由度量接口
支持 K-最短路径算法的度量指标
"""
from abc import ABC, abstractmethod
import networkx as nx
from typing import List, Dict, Any

class RoutingMetric(ABC):
    """
    路由度量抽象基类
    支持路径级和边级的度量计算
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        返回度量指标的名称。
        例如: "UEC", "HopCount", "Distance"
        """
        pass

    @abstractmethod
    def calculate_path_cost(self, graph: nx.Graph, path: List[int]) -> tuple[float, Dict[int, float]]:
        """
        计算完整路径的总成本和节点损耗。
        用于 KSP 算法对候选路径进行精确排序。

        Args:
            graph: NetworkX 图对象
            path: 节点路径列表，如 [0, 2, 4, 7]

        Returns:
            tuple[float, Dict[int, float]]: (总成本, 节点损耗字典)
                总成本: 路径总成本 (越小越好)
                节点损耗字典: 键为节点ID，值为该节点的损耗
        """
        pass

    @abstractmethod
    def get_edge_weight(self, u: int, v: int, edge_attr: Dict[str, Any]) -> float:
        """
        计算单条边的权重。
        用于 Dijkstra/KSP 搜索时的边权重，必须为非负值。

        Args:
            u: 起始节点
            v: 结束节点
            edge_attr: 边的属性字典

        Returns:
            float: 边的权重 (越小越好，非负)
        """
        pass
