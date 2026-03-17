import networkx as nx
import math
import random

from core.Node import QuantumNode
from core.Link import QuantumLink
from core.Request import QuantumRequest, RequestStatus
from utils.Constants import RES_CONFIG, TOPO_CONFIG


class QuantumNetworkManager:
    """
    基于 NetworkX 和 Waxman 模型的量子网络管理器
    """

    def __init__(self):
        self.width = TOPO_CONFIG["WIDTH"]
        self.height = TOPO_CONFIG["HEIGHT"]

        self.graph = nx.Graph()

    def generate_waxman_topology(self):
        """
        TOPO_CONFIG 中配置区域宽度、区域高度、节点数、Waxman参数
        """
        self.graph.clear()

        alpha = TOPO_CONFIG["ALPHA"]
        beta = TOPO_CONFIG["BETA"]

        L_max = math.hypot(self.width, self.height)

        # ---------------------------------------------------
        # 第一阶段：生成物理拓扑
        # ---------------------------------------------------

        # 计算最小节点间距50/sqrt(n),确保节点不会过于密集
        min_distance = 50 / math.sqrt(TOPO_CONFIG['NUM_NODES'])

        # 1. 生成节点（保证最小间距）
        for i in range(TOPO_CONFIG["NUM_NODES"]):
            max_attempts = 1000  # 防止无限循环
            attempts = 0

            while attempts < max_attempts:
                # 随机生成候选位置
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)

                # 检查与已存在节点的最小距离
                valid_position = True
                for existing_node_id in self.graph.nodes():
                    existing_node = self.graph.nodes[existing_node_id]['object']
                    distance = math.hypot(x - existing_node.x_coord, y - existing_node.y_coord)
                    if distance < min_distance:
                        valid_position = False
                        break

                if valid_position:
                    # 位置有效，创建节点
                    q_node = QuantumNode(
                        node_id=i,
                        x_coord=x,
                        y_coord=y,
                        max_rate=0,
                        max_memory=0
                    )
                    self.graph.add_node(i, object=q_node, pos=(x, y))
                    break

                attempts += 1

            if attempts >= max_attempts:
                print(f"Warning: Could not find valid position for node {i} after {max_attempts} attempts")
                # 即使找不到理想位置，也要创建节点（使用最后一次尝试的位置）
                q_node = QuantumNode(
                    node_id=i,
                    x_coord=x,
                    y_coord=y,
                    max_rate=0,
                    max_memory=0
                )
                self.graph.add_node(i, object=q_node, pos=(x, y))

        # 2. 生成链路
        for u in range(TOPO_CONFIG["NUM_NODES"]):
            for v in range(u + 1, TOPO_CONFIG["NUM_NODES"]):
                node_u = self.graph.nodes[u]['object']
                node_v = self.graph.nodes[v]['object']

                dist = math.hypot(node_u.x_coord - node_v.x_coord, node_u.y_coord - node_v.y_coord)

                # Waxman 概率公式
                prob = alpha * math.exp(-dist / (beta * L_max))

                if random.random() < prob:
                    # 创建链路 (会自动计算 链路衰减因子、链路保真度)
                    link = QuantumLink(u=u, v=v, length=dist)
                    self.graph.add_edge(u, v, object=link, weight=dist)

        # 连通性检查与递归重试
        if not nx.is_connected(self.graph):
            print("Generated graph is not connected, retrying...")
            self.generate_waxman_topology()
            return

        # ---------------------------------------------------
        # 第二阶段：资源异构分配
        # ---------------------------------------------------
        self._assign_heterogeneous_resources(TOPO_CONFIG["NUM_NODES"])

        print(f"Topology Generated: {TOPO_CONFIG['NUM_NODES']} nodes, {self.graph.number_of_edges()} links.")

    def _assign_heterogeneous_resources(self, num_nodes: int):
        """根据节点的度 (Degree) 分配 HIGH/LOW 资源"""
        degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)

        # Top 20% 为核心节点
        high_count = max(1, int(num_nodes * 0.2))
        high_priority_ids = {nid for nid, _ in degrees[:high_count]}

        for node_id, attrs in self.graph.nodes(data=True):
            q_node = attrs['object']
            if node_id in high_priority_ids:
                config = RES_CONFIG["HIGH"]
            else:
                config = RES_CONFIG["LOW"]

            q_node.max_rate = config["rate"]
            q_node.max_memory = config["memory"]

            q_node.reset()

    def get_node_object(self, node_id: int) -> QuantumNode:
        return self.graph.nodes[node_id]['object']

    def get_link_object(self, u: int, v: int) -> QuantumLink:
        """从 NetworkX 边属性中获取 Link 对象"""
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['object']
        return None

    def get_average_degree(self) -> float:
        """
        计算全网平均度数
        利用握手定理: Sum(Degrees) = 2 * Edge_Count
        """
        num_nodes = self.graph.number_of_nodes()

        if num_nodes == 0:
            return 0.0

        num_edges = self.graph.number_of_edges()

        # 平均度数 = (2 * 边数) / 节点数
        return (2 * num_edges) / num_nodes

    def attempt_allocate_resource(self, request: QuantumRequest, path_index: int = 0) -> tuple[int, int]:
        """
        尝试分配资源
        Args:
            request: QuantumRequest 对象
            path_index: 路径索引
        Returns:
            int: 本次成功分配的纠缠对数量 (若为0则表示失败)
        """
        # 数据准备 (Data Preparation)
        path_nodes = request.paths.get(path_index)
        path_eff_data = request.UEC_eff.get(path_index)

        if any(c == float('inf') for c in path_eff_data.values()):
            return 0, 0
        # --- 阶段 A: 构建单位消耗图 (Build Unit Cost Maps) ---
        unit_demand_Re = path_eff_data
        unit_demand_M = {node: 0.0 for node in path_nodes}
        # 遍历路径上的链路，累加双端消耗
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            l_eff = path_eff_data.get(u, 0.0)

            if l_eff == 0: continue  # 无损耗则跳过

            # 关键: 链路 u->v 的消耗，u 和 v 都要承担
            unit_demand_M[u] += l_eff
            unit_demand_M[v] += l_eff
        # --- 阶段 B: 全局瓶颈探测 (Bottleneck Detection) ---
        path_capacity = float('inf')
        for node_id in path_nodes:
            node = self.graph.nodes[node_id]['object']
            # 策略: 优先计算存储瓶颈 (M)
            cost_m = unit_demand_M.get(node_id, 0.0)
            limit_m = float('inf')
            avail_m = node.max_memory - node.current_memory
            if cost_m > 0:
                if avail_m <= 0:
                    limit_m = 0
                else:
                    limit_m = int(avail_m // cost_m)
            # 剪枝: 若存储已经卡死，直接退出
            if limit_m == 0:
                path_capacity = 0
                break
            # 反向检查速率资源 (Re)
            cost_re = unit_demand_Re.get(node_id, 0.0)
            node_limit = limit_m  # 默认瓶颈就是 M
            if cost_re > 0:
                avail_re = node.max_rate - node.current_rate
                if limit_m * cost_re > avail_re:
                    # 生成受限场景
                    if avail_re <= 0:
                        limit_re = 0
                    else:
                        limit_re = int(avail_re // cost_re)
                    node_limit = limit_re
            # 更新路径短板
            if node_limit < path_capacity:
                path_capacity = node_limit
                if path_capacity == 0: break
        # --- 阶段 C: 决策 (Decision) ---
        allocation = min(request.remaining_pairs, int(path_capacity))
        if allocation <= 0:
            return 0, 0
        # --- 阶段 D: 执行扣款 (Execution) ---
        total_physical_cost = 0  # 物理成本累加器
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            l_eff = path_eff_data.get(u, 0.0)
            if l_eff == 0: continue
            link_load = math.ceil(l_eff * allocation)
            total_physical_cost += link_load
            node_u = self.graph.nodes[u]['object']
            node_v = self.graph.nodes[v]['object']
            # u 是发送端: 消耗 Re (生成) + M (存储)
            node_u.consume(link_load, link_load)
            # v 是接收端: 消耗 M (存储)
            node_v.consume(0, link_load)
        # --- 阶段 E: 更新状态 (State Update) ---
        request.remaining_pairs -= allocation

        if request.remaining_pairs <= 0:
            request.status = RequestStatus.ACCEPTED
        else:
            request.status = RequestStatus.ROUTED
        return allocation, total_physical_cost

    def reset_resources(self):
        """
        重置整个网络中所有节点的资源占用状态
        """
        # 遍历 NetworkX 图中的所有节点 ID
        for node_id in self.graph.nodes:
            # 获取实际的 Node 对象
            node = self.graph.nodes[node_id]['object']
            node.reset()

        # ——————————————————RL依赖代码————————————————————————————

    def attempt_allocate_resource_rl(self, request: 'QuantumRequest', path_index: int, attempt_amount: int) -> tuple[bool, int]:
        """
        专为 RL 设计的资源分配方法 (All-or-Nothing)
        智能体提议一个分配量，若网络资源足以支撑该请求，则全额扣除并返回 True 和 物理成本；
        若任何一个节点资源不足，则全盘拒绝 (Early Stopping)，返回 False 和 0。

        Args:
            request: QuantumRequest 对象
            path_index: 路径索引
            attempt_amount: 智能体提议分配的纠缠对数量

        Returns:
            tuple[bool, int]: (是否分配成功, 消耗的物理总资源)
        """
        if attempt_amount <= 0:
            return False, 0

        path_nodes = request.paths.get(path_index)
        path_eff_data = request.UEC_eff.get(path_index)

        if not path_nodes or not path_eff_data:
            return False, 0

        if any(c == float('inf') for c in path_eff_data.values()):
            return False, 0

        # --- 阶段 A & B: 边计算边探测 (Dynamic Demand Calculation & Early Stopping) ---
        required_Re = {node: 0 for node in path_nodes}
        required_M = {node: 0 for node in path_nodes}
        total_physical_cost = 0

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            l_eff = path_eff_data.get(u, 0.0)

            if l_eff == 0: continue

            # 1. 计算当前边的物理消耗
            link_load = math.ceil(l_eff * attempt_amount)
            total_physical_cost += link_load

            # 2. 累加到节点的总需求中
            required_Re[u] += link_load
            required_M[u] += link_load
            required_M[v] += link_load

            # 3. 实时溢出探测 (Early Stopping): 只要刚加上就超了，立刻宣告分配失败！
            node_u = self.graph.nodes[u]['object']
            if required_Re[u] > (node_u.max_rate - node_u.current_rate) or required_M[u] > (node_u.max_memory - node_u.current_memory):
                return False, 0

            node_v = self.graph.nodes[v]['object']
            if required_M[v] > (node_v.max_memory - node_v.current_memory):
                return False, 0

        # --- 阶段 C: 执行扣款 (Execution) ---
        # 能走到这里，说明整条路径的需求算完了，且没有任何节点溢出
        for node_id in path_nodes:
            node = self.graph.nodes[node_id]['object']
            req_re = required_Re[node_id]
            req_m = required_M[node_id]

            if req_re > 0 or req_m > 0:
                node.consume(req_re, req_m)

        # --- 阶段 D: 更新状态 (State Update) ---
        request.remaining_pairs -= attempt_amount
        if request.remaining_pairs <= 0:
            request.status = RequestStatus.ACCEPTED
        else:
            request.status = RequestStatus.ROUTED
        # 成功分配！返回 True 和消耗的底层物理对数
        return True, total_physical_cost

        # ——————————————————测试方法代码————————————————————————————

    # 自定义网络拓扑添加节点和连接
    def add_node(self, node_id: int):
        q_node = QuantumNode(
            node_id=node_id,
            x_coord=random.uniform(0, self.width),
            y_coord=random.uniform(0, self.height),
            max_rate=500,
            max_memory=50
        )
        self.graph.add_node(node_id, object=q_node)
        return q_node

    def add_custom_connection(self, u: int, v: int, attenuation: float, fidelity: float, length: float = 1.0):
        """
        添加自定义链路连接，允许手动指定衰减因子和保真度

        Args:
            u: 起始节点ID
            v: 结束节点ID
            attenuation: 手动指定的衰减因子
            fidelity: 手动指定的保真度
            length: 链路长度（用于标识，默认1.0）

        Returns:
            QuantumLink: 创建的链路对象
        """
        # 创建 QuantumLink 对象（会触发自动计算）
        link = QuantumLink(u=u, v=v, length=length)

        # 手动覆盖自动计算的值
        link.attenuation = attenuation
        link.fidelity = fidelity

        # 添加到图中
        self.graph.add_edge(u, v, object=link, weight=length)

        return link
