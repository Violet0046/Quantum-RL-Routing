import copy
import time
import csv
from tqdm import tqdm
from typing import List, Dict
from typing import Type
from routing.metrics.BaseMetric import RoutingMetric
from core.Network import QuantumNetworkManager
from routing.KSP import QuantumRouter
from core.Request import QuantumRequest, RequestStatus
from utils.RequestGenerator import RequestGenerator
from utils.Constants import SCHEDULER_CONFIG, SIMULATION_CONFIG
from utils.PathCandidate import PathCandidate

# ==========================================
# 2. Simulator 类
# ==========================================
class QuantumSimulator:
    def __init__(self, metric_class: Type[RoutingMetric], external_network: QuantumNetworkManager = None):
        """
        初始化仿真器
        """
        metric_instance = metric_class()
        # 仿真配置
        self.max_slots = SIMULATION_CONFIG.get('MAX_SLOTS', 100)
        # self.requests_per_slot = SIMULATION_CONFIG.get('REQUESTS_PER_SLOT', 5)
        self.lam = SIMULATION_CONFIG.get('LAM', 5)  # 泊松分布的期望值 Lambda (即平均每时隙产生的请求数)
        # 1. 初始化网络与拓扑
        if external_network is not None:
            # 若外部传入了网络，使用深拷贝！
            self.network = copy.deepcopy(external_network)
            print("Simulator: Loaded EXTERNAL topology (Deep Copy).")
        else:
            self.network = QuantumNetworkManager()
            self.network.generate_waxman_topology()
            print("Simulator: Generated INTERNAL Waxman topology.")
        # 2. 初始化路由
        self.router = QuantumRouter(self.network.graph, metric_instance)

        # 3. 初始化流量生成器
        self.generator = RequestGenerator(self.network)

        # 4. 核心队列管理
        # pending_requests 存储所有状态为 ROUTED 但 remaining_pairs > 0 的请求
        self.pending_requests: List[QuantumRequest] = []

        # 统计列表
        self.completed_requests: List[QuantumRequest] = []
        self.blocked_requests: List[QuantumRequest] = []  # 保真度不达标或超时

        # 统计指标
        self.current_slot = 0
        self.total_requests = 0
        # 总共生成的比特数 (纠缠对数量)
        self.total_allocated_bits = 0
        self.total_consumed_pairs = 0  # 总消耗
        # 对比实验设置,use_preloaded = True时注入预生成好的请求
        self.use_preloaded = False
        self.preloaded_pool = {}

        # CSV 记录相关
        # 文件名格式: sim_UEC_20260127_233000.csv
        algo_name = metric_instance.name
        timestamp = time.strftime('%Y%m%d')  # _%H%M%S 时分秒
        self.csv_filename = f"sim_{algo_name}_{timestamp}.csv"

        # 初始化 CSV 文件并写入表头
        headers = [
            "Slot", "Total_Requests", "Completed", "Pending",
            "Blocked", "Success_Rate", "Throughput_Avg", "Efficiency_Avg"
        ]

        # 'w' 模式创建新文件，newline='' 防止空行
        with open(self.csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def run(self):
        """
        执行仿真主循环
        """
        print(f"Simulation started for {self.max_slots} slots...")
        start_time = time.time()
        # 打印进度
        for slot in tqdm(range(self.max_slots), desc="Simulating", dynamic_ncols=True, unit="slot"):
            self.current_slot = slot
            self._process_time_slot(slot)
            if (slot + 1) % 10 == 0:
                self._record_metrics(slot)

        elapsed = time.time() - start_time
        print(f"\n>>> Simulation Finished in {elapsed:.2f} seconds.")
        self._print_summary()

    def _process_time_slot(self, slot_index: int):
        """
        处理单个时隙的核心逻辑
        """
        # 超时请求阈值
        max_wait = SIMULATION_CONFIG.get("MAX_WAIT", 50)
        alive_requests = []
        # =========================================================
        # Phase 1: 资源复位 (Reset)
        # =========================================================
        # 清空所有节点的 current_rate 和 current_memory
        self.network.reset_resources()

        # =========================================================
        # Phase 2: 老化 (Aging)
        # =========================================================
        # 对于上一轮遗留的 ROUTED 请求，等待时隙 +1
        # 这样在后续计算分数时，它们的优先级会提升
        for req in self.pending_requests:
            req.waiting_slots += 1
            if req.waiting_slots > max_wait:
                req.status = RequestStatus.BLOCKED_EXPIRED
                self.blocked_requests.append(req)
            else:
                alive_requests.append(req)
        self.pending_requests = alive_requests
        # =========================================================
        # Phase 3: 新流量接入 (Arrivals)
        # =========================================================
        new_requests = []
        if self.use_preloaded:
            original_requests = self.preloaded_pool.get(slot_index, [])

            if original_requests:
                new_requests = copy.deepcopy(original_requests)  # 深拷贝原始请求
        else:
            # new_requests = self.generator.generate_multiple_requests(self.requests_per_slot, start_id = self.total_requests)
            new_requests = self.generator.generate_requests_poisson(self.lam, start_id=self.total_requests)  # 泊松分布生成请求
        self.total_requests += len(new_requests)

        active_new_requests = []
        for req in new_requests:
            self.router.k_shortest_paths(req)

            if self.router.metric.name != 'UEC':
                req.compute_uec_eff(self.network.graph)

            if req.status == RequestStatus.ROUTED:
                active_new_requests.append(req)
            else:
                self.blocked_requests.append(req)

        self.pending_requests.extend(active_new_requests)
        # =========================================================
        # Phase 4: 调度与分配 (Scheduling & Allocation)
        # =========================================================
        # 构建调度池 PathCandidate Pool
        scheduling_pool = []

        # 获取调度权重因子 (Time Weight)
        alpha = SCHEDULER_CONFIG.get("ALPHA", 0.0)

        for req in self.pending_requests:
            # 遍历该请求的所有 K 条备选路径
            # candidate_costs 是一个字典 {path_index: cost}
            for k_idx, cost in req.candidate_costs.items():
                if cost == float('inf'): continue  # 跳过无效路径

                # 计算排序分数: Score = Cost - (Alpha * Waiting)
                score = cost - (alpha * req.waiting_slots)

                # 创建候选对象 (不包含 redundant 的 cost)
                candidate = PathCandidate(
                    sort_score=score,
                    request=req,
                    path_index=k_idx
                )
                scheduling_pool.append(candidate)

        # 对所有请求的所有路径进行排序 (按照Score从小到大排序)
        scheduling_pool.sort()

        # 开始分配
        for cand in scheduling_pool:
            req = cand.request

            # 如果请求已经分配成功，则跳过
            if req.status == RequestStatus.ACCEPTED or req.remaining_pairs <= 0:
                continue

            # 尝试在该路径上分配资源
            # attempt_allocate_resource 返回本次分配的纠缠对数量
            # 并内部扣除节点的 current_rate/memory
            allocated_bits, physical_cost = self.network.attempt_allocate_resource(req, cand.path_index)
            self.total_allocated_bits += allocated_bits
            self.total_consumed_pairs += physical_cost
            # 检查分配后状态
            if req.remaining_pairs <= 0:
                req.status = RequestStatus.ACCEPTED

        # =========================================================
        # Phase 5: 清理 (Cleanup)
        # =========================================================
        # 将已完成的请求移出 pending_requests，放入 completed_requests
        # 未完成的请求保留在 pending_requests 中，进入下一个时隙

        next_slot_pending = []
        for req in self.pending_requests:
            if req.status == RequestStatus.ACCEPTED:
                self.completed_requests.append(req)
            else:
                next_slot_pending.append(req)

        self.pending_requests = next_slot_pending

    def set_preloaded_requests(self, request_pool: Dict[int, List[QuantumRequest]]):
        """
        允许外部注入预先生成好的请求
        """
        self.preloaded_pool = request_pool
        self.use_preloaded = True

    def _record_metrics(self, slot_index):
        """
        将当前时隙的统计指标写入 CSV
        """
        # 1. 计算当前瞬时/累计指标
        duration = slot_index + 1

        # 成功率
        sr = len(self.completed_requests) / self.total_requests if self.total_requests > 0 else 0

        # 平均吞吐量
        tp = self.total_allocated_bits / duration

        # 资源效率
        eff = self.total_allocated_bits / self.total_consumed_pairs if self.total_consumed_pairs > 0 else 0

        # 2. 准备数据行
        row = [
            slot_index,
            self.total_requests,
            len(self.completed_requests),
            len(self.pending_requests),
            len(self.blocked_requests),  # 保真度不达标或超时
            f"{sr:.4f}",  # 保留4位小数
            f"{tp:.4f}",
            f"{eff:.6f}"
        ]

        # 3. 写入文件 (使用 'a' 追加模式)
        with open(self.csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _print_summary(self):
        print("\n=== Simulation Summary ===")
        print(f"Total Requests Generated: {self.total_requests}")
        print(f"Completed: {len(self.completed_requests)}")
        print(f"Blocked (Topo/Fidelity): {len(self.blocked_requests)}")
        print(f"Pending (Unfinished): {len(self.pending_requests)}")

        # 成功率分析
        if self.total_requests > 0:
            success_rate = len(self.completed_requests) / self.total_requests
            print(f"Success Rate: {success_rate:.2%}")

        # 吞吐量分析
        total_duration = self.current_slot + 1
        if total_duration > 0:
            # 计算平均每个时隙产生的比特数
            throughput = self.total_allocated_bits / total_duration
            print(f"Throughput (Avg):    {throughput:.4f} bits/slot (pairs/slot)")
            # 总数
            print(f"Total Bits Gen.:     {self.total_allocated_bits}")

        # 时延分析 (基于 waiting_slots)
        if self.completed_requests:
            avg_wait = sum(req.waiting_slots for req in self.completed_requests) / len(self.completed_requests)
            max_wait = max(req.waiting_slots for req in self.completed_requests)
            print(f"Avg Waiting Time:    {avg_wait:.2f} slots")
            print(f"Max Waiting Time:    {max_wait} slots")

        # 资源利用率分析
        if self.total_consumed_pairs > 0:
            efficiency = self.total_allocated_bits / self.total_consumed_pairs
            print(f"  - Efficiency:      {efficiency:.4f} bits/pair")
        else:
            print(f"  - Efficiency:      N/A (No resources consumed)")

