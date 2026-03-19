import networkx as nx
import math
from typing import List, Dict, Any
from .BaseMetric import RoutingMetric
from utils.Constants import P_SWAP_DEFAULT
from core.Link import QuantumLink

class Q_CASTMetric(RoutingMetric):

    @property
    def name(self) -> str:
        return "Q-CAST"

    def calculate_path_cost(self, graph: nx.Graph, path: List[int]) -> float:
        """
        计算路径的期望吞吐量 (EXT - Expected Throughput)
        对应论文公式 (2) 和 Appendix A.2.1
        核心思想: E[T] = Sum( P(Path_Width >= k) ) for k in 1..max_width
        """
        if len(path) < 2:
            return 0.0

        # 1. 获取路径上所有边的物理属性
        edges_props = []
        min_path_width = float('inf') # 路径的物理瓶颈宽度

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = graph[u][v]
            
            # 获取该边的物理宽度 (User logic: min(max_memory))
            # 注意：这里假设 graph 节点存了 max_memory，或者 edge_data 里有 'width'
            # 按照你之前的逻辑，我们取边的 'width' 属性，如果没有则用内存计算
            width = edge_data.get('width')
            if width is None:
                # 回退策略：如果没有显式宽度，用节点内存估算
                w_u = graph.nodes[u].get('max_memory', 50) # 默认值防报错
                w_v = graph.nodes[v].get('max_memory', 50)
                width = min(w_u, w_v)

            # 获取链路成功率
            link: QuantumLink | None = edge_data.get('object', None)
            if link:
                p = link.attenuation
            else:
                p = 0.9

            edges_props.append({'w': width, 'p': p})
            if width < min_path_width:
                min_path_width = width

        if min_path_width == 0:
            return 0.0

        # 2. 计算概率乘积
        # E_t = Sum_{k=1 to W} [ Product_{edges} (P(edge_success >= k)) ]
        expected_throughput = 0.0
        
        # 预计算每一条边的 "至少成功 k 个" 的概率
        # 优化：我们只需要计算 k 从 1 到 min_path_width
        # 因为 k > min_path_width 时，概率一定为 0
        for k in range(1, int(min_path_width) + 1):
            prob_path_ge_k = 1.0 # P(Path >= k)
            
            for edge in edges_props:
                w = edge['w']
                p = edge['p']
                
                # 计算这条边: P(Binomial(w, p) >= k)
                # P(X >= k) = Sum_{j=k}^{w} C(w, j) * p^j * (1-p)^(w-j)
                prob_edge_ge_k = 0.0
                for j in range(k, int(w) + 1):
                    # 二项分布概率公式
                    prob = math.comb(int(w), j) * (p ** j) * ((1 - p) ** (int(w) - j))
                    prob_edge_ge_k += prob
                
                prob_path_ge_k *= prob_edge_ge_k
                
                # 剪枝：如果概率太小，后面不仅乘起来也是0，直接break
                if prob_path_ge_k < 1e-6:
                    break
            
            expected_throughput += prob_path_ge_k

        # 3. 考虑交换成功率 q (Swapping Success Rate)
        # 论文 Eq 2: E_I = q^h * ...
        q_swap = P_SWAP_DEFAULT
        num_swaps = len(path) - 2 # 中间节点数
        if num_swaps > 0:
             expected_throughput *= (q_swap ** num_swaps)

        # 返回的是期望吞吐量 (越大越好) 但是其它度量指标均为越小越好，为了统一，返回负值
        return -expected_throughput

    def get_edge_weight(self, u: int, v: int, edge_attr: Dict[str, Any]) -> float:
        """
        Q-CAST (BotCap Metric) 权重计算
        返回元组: (-Width, 1/P_link)
        NetworkX 的 Dijkstra 如果支持元组比较，会优先比较第一个元素 (Width 越大，-Width 越小)
        如果 Width 相同，再比较第二个元素 (P_link 越大，Cost 越小)
        """       
        # 获取概率 P
        p = edge_attr.get('p_link', 1e-9) # 防止除零
        if p <= 0: p = 1e-9
        
        # 返回CR
        return 1.0 / p