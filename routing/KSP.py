"""
K-最短路径路由算法
"""
import networkx as nx
from typing import List, Dict, Any
from routing.metrics.BaseMetric import RoutingMetric
from core.Request import QuantumRequest, RequestStatus
from utils.Constants import KSP_CONFIG

class QuantumRouter:
    """
    可以使用不同的路由度量指标
    """

    def __init__(self, graph: nx.Graph, metric: RoutingMetric):
        """
        初始化路由器
        Args:
            graph: NetworkX 图对象
            metric: 路由度量指标
        """
        self.graph = graph
        self.metric = metric

        self._path_cache = {}

    def k_shortest_paths(self, request: QuantumRequest, k: int = None):
        """
        计算 K-最短路径并将节点损耗信息存储到请求中

        Args:
            request: QuantumRequest 对象，包含源节点和目标节点
            k: 返回的路径数量
        """
        if k is None:
            k = KSP_CONFIG.get("K", 3)

        source = request.source
        target = request.target
        fidelity_threshold = request.fidelity_threshold

        if self.metric.name != 'UEC':
            cache_key = (source, target)
        else:
            cache_key = (source, target, fidelity_threshold)

        # 检查缓存
        if cache_key in self._path_cache:
            if self._path_cache[cache_key] is None:
                request.mark_as_blocked_topology()
                return
            all_cached_paths = self._path_cache[cache_key]
            if len(all_cached_paths) >= k:
                paths_to_use = all_cached_paths[:k]
                self._apply_cached_paths(request, paths_to_use, k)
                return
            else: pass                

        try:
            # 计算物理最短跳数 (无权图 BFS)
            min_hops = nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            request.mark_as_blocked_topology()   #物理上不可达
            self._path_cache[cache_key] = None
            return

        # 悲观策略:会让 Metric 对保真度要求更严格
        estimated_hops = min_hops + 1
        # 如果 Metric 支持上下文注入 (即 UECMetric)，则传入当前请求的约束
        if hasattr(self.metric, 'set_context'):
            self.metric.set_context(fidelity_threshold , estimated_hops)
        # 1. 使用度量指标的边权重生成候选路径
        candidate_generator = nx.shortest_simple_paths(
            self.graph,
            source,
            target,
            weight=lambda u, v, attr: self.metric.get_edge_weight(u, v, attr)
        )
        potential_paths = []

        # 2. 获取候选路径池 (取 k*3 个，确保能根据UEC度量指标找到最优路径)
        pool_size = k * 3
        for _ in range(pool_size):
            try:
                path = next(candidate_generator)
                cost_result = self.metric.calculate_path_cost(self.graph, path)

                # 处理不同度量返回值的兼容性
                if isinstance(cost_result, tuple):
                    cost, node_losses = cost_result
                else:
                    cost = cost_result
                    node_losses = None  # 标记为无损耗信息

                potential_paths.append({
                    'path': path,
                    'cost': cost,
                    'node_losses': node_losses  # 存储节点损耗信息，None表示无损耗信息
                })
            except StopIteration:
                break

        # 3. 根据精确成本排序并返回前 k 个
        sorted_paths = sorted(potential_paths, key=lambda x: x['cost'])
        result = sorted_paths[:k]

        # 4. 将节点损耗信息存储到请求中
        self._update_request_routing_results(request, result, k)
        
        # 5. 缓存结果（不包含 node_losses 以节省内存）
        cached_result = [{'path': p['path'], 'cost': p['cost']} for p in sorted_paths]
        self._path_cache[cache_key] = cached_result


    def clear_cache(self):
        """
        当网络拓扑发生改变时，调用此方法清空缓存
        """
        self._path_cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。

        Returns:
            Dict: 包含缓存统计信息的字典
                - 'total_entries': 缓存条目总数
                - 'cache_keys': 缓存键列表
        """
        return {
            'total_entries': len(self._path_cache),
            'cache_keys': list(self._path_cache.keys())
        }

    def find_best_path(self, request: QuantumRequest):
        """
        找到最佳路径
        Args:
            request: QuantumRequest 对象
        """
        self.k_shortest_paths(request, k=1)
        return

    def _update_request_routing_results(self, request: QuantumRequest, result: List[Dict[str, Any]], k: int):
        """
        将路由计算的结果（节点损耗详情和路径总成本）更新到 Request 对象中。
        """
        request.paths.clear()
        request.UEC_eff.clear()
        if self.metric.name != 'UEC':
            request.status = RequestStatus.ROUTED
            for i, path_info in enumerate(result):
                if i >= k:
                    break
                request.candidate_costs[i] = path_info['cost']
                request.paths[i] = path_info['path']
            return

        for i, path_info in enumerate(result):
            if i >= k:
                break
            # node_losses -> 完整的节点损耗字典 {node_id: loss}
            node_losses = path_info.get('node_losses')
            if i == 0 and path_info['cost'] == float('inf'): 
                request.mark_as_blocked_fidelity()
                return
            request.candidate_costs[i] = path_info['cost']
            request.UEC_eff[i] = node_losses
            request.paths[i] = path_info['path']
       
        request.status = RequestStatus.ROUTED

    def _apply_cached_paths(self, request: QuantumRequest, cached_paths: List[Dict[str, Any]], k: int):
        """
        从缓存中恢复信息到请求

        Args:
            request: QuantumRequest 对象
            cached_paths: 缓存的路径列表
            k: 路径数量
        """
        request.paths.clear()
        request.UEC_eff.clear()
        if self.metric.name != 'UEC':
            request.status = RequestStatus.ROUTED
            for i, path_info in enumerate(cached_paths):
                if i >= k:
                    break
                request.candidate_costs[i] = path_info['cost']
                request.paths[i] = path_info['path']
            return

        if hasattr(self.metric, 'set_context'):
            self.metric.set_context(request.fidelity_threshold, 1)

        # 重新计算
        for i, path_info in enumerate(cached_paths):
            if i >= k:
                break
            path = path_info['path']
            #计算成本
            cost, node_losses = self.metric.calculate_path_cost(self.graph, path)
            if i == 0 and cost == float('inf'): 
                request.mark_as_blocked_fidelity()
                return
            request.candidate_costs[i] = cost
            request.UEC_eff[i] = node_losses
            request.paths[i] = path
        request.status = RequestStatus.ROUTED