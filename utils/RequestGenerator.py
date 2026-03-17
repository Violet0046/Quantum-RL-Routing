"""
请求生成器
自动生成量子网络请求
"""
import random
from typing import List, Optional
from core.Request import QuantumRequest
from core.Network import QuantumNetworkManager
from utils.Constants import REQUEST_CONFIG
import numpy as np


class RequestGenerator:
    """
    量子网络请求生成器
    用于自动生成符合网络拓扑和参数约束的请求
    """

    def __init__(self, network: QuantumNetworkManager):
        """
        初始化请求生成器

        Args:
            network: 量子网络管理器，用于获取网络拓扑信息
        """
        self.network = network
        # 暂时手动获取节点ID列表
        self.node_ids = sorted(list(network.graph.nodes()))

        if len(self.node_ids) < 2:
            raise ValueError("网络中至少需要2个节点才能生成请求")

    def generate_single_request(self, request_id: int,
                               source: Optional[int] = None,
                               target: Optional[int] = None) -> QuantumRequest:
        """
        生成单个请求

        Args:
            request_id: 请求ID
            source: 源节点ID,如果为None则随机选择。
            target: 目标节点ID,如果为None则随机选择(确保与源节点不同)

        Returns:
            QuantumRequest: 生成的请求对象
        """
        # 选择源节点
        if source is None:
            source = random.choice(self.node_ids)
        elif source not in self.node_ids:
            raise ValueError(f"源节点 {source} 不存在于网络中")

        # 选择目标节点（确保与源节点不同）
        available_targets = [node for node in self.node_ids if node != source]
        if target is None:
            target = random.choice(available_targets)
        elif target not in self.node_ids:
            raise ValueError(f"目标节点 {target} 不存在于网络中")
        elif target == source:
            raise ValueError("源节点和目标节点不能相同")

        # 生成请求对数量
        num_pairs = random.randint(
            REQUEST_CONFIG["MIN_NUM_PAIRS"],
            REQUEST_CONFIG["MAX_NUM_PAIRS"]
        )

        # 生成保真度阈值
        fidelity_threshold = random.choices(
            population = REQUEST_CONFIG["FIDELITY_LEVELS"],
            weights = REQUEST_CONFIG.get("FIDELITY_PROBABILITIES"), # 若没配概率，则默认均匀分布
            k=1
        )[0]

        # 创建请求
        request = QuantumRequest(
            request_id=request_id,
            source=source,
            target=target,
            num_pairs=num_pairs,
            fidelity_threshold=fidelity_threshold
        )

        return request

    def generate_multiple_requests(self, num_requests: int,
                                  start_id: int = 0) -> List[QuantumRequest]:
        """
        生成多个请求

        Args:
            num_requests: 要生成的请求数量
            start_id: 起始请求ID

        Returns:
            List[QuantumRequest]: 生成的请求列表
        """
        requests = []
        for i in range(num_requests):
            request = self.generate_single_request(request_id=start_id + i)
            requests.append(request)

        return requests

    def generate_requests_poisson(self, lam: float, start_id: int = 0) -> List[QuantumRequest]:
        """
        基于泊松分布生成请求列表 (模拟真实突发流量)

        Args:
            lam (float): 泊松分布的期望值 Lambda (即平均每时隙产生的请求数)
            start_id (int): 起始ID

        Returns:
            List[QuantumRequest]: 生成的请求列表 (数量是不确定的,可能为0,也可能很大)
        """
        # 1. 核心步骤：根据期望值 lam，随机生成“这一次”的实际数量
        # 例如 lam=5，这里可能返回 3, 5, 8, 12, 1 等
        actual_num_requests = np.random.poisson(lam=lam)

        # 2. 如果随机到 0，直接返回空列表 (模拟网络空闲)
        if actual_num_requests == 0:
            return []

        # 3. 复用你已有的固定数量生成逻辑
        return self.generate_multiple_requests(num_requests=actual_num_requests, start_id=start_id)
# 便捷函数
def create_request_generator(network: QuantumNetworkManager) -> RequestGenerator:
    """
    创建请求生成器的便捷函数

    Args:
        network: 量子网络管理器

    Returns:
        RequestGenerator: 请求生成器实例
    """
    return RequestGenerator(network)