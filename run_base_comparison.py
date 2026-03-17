import time

from core.Network import QuantumNetworkManager
from scheduler.GreedyScheduler import QuantumSimulator
from utils.RequestGenerator import RequestGenerator
from routing.metrics.UEC import UECMetric
from routing.metrics.MinHops import MinHopsMetric
from utils.Constants import SIMULATION_CONFIG
from routing.metrics.CR import CRMetric

import random
import numpy as np
# ==========================================
# 1. 环境准备
# ==========================================
def generate_benchmark_environment():
    """
    生成:
    1. 固定的物理拓扑
    2. 固定的请求数据集 (覆盖所有时隙)
    """
    print(f"[{time.strftime('%H:%M:%S')}] >>> Initializing Benchmark Environment...")
    # 固定拓扑种子，确保拓扑结构永远一致
    topo_seed = 202433851
    random.seed(topo_seed)
    np.random.seed(topo_seed)

    # 1. 生成基准拓扑
    base_network = QuantumNetworkManager()
    base_network.generate_waxman_topology()
    print(f"  - Topology Created: {len(base_network.graph.nodes)} nodes, {len(base_network.graph.edges)} links.")

    # 2. 生成基准请求集
    # 指定测试集种子
    TEST_REQ_SEED = 8888
    random.seed(TEST_REQ_SEED)
    np.random.seed(TEST_REQ_SEED)
    generator = RequestGenerator(base_network)
    request_dataset = {}
    total_generated = 0
    
    max_slots = SIMULATION_CONFIG["MAX_SLOTS"]
    #req_per_slot = SIMULATION_CONFIG["REQUESTS_PER_SLOT"]
    #print(f"  - Generating  requests for {max_slots} slots (Load={req_per_slot})...")

    lam = SIMULATION_CONFIG["LAM"]
    print(f"  - Generating poisson requests for {max_slots} slots (Load={lam})...")
    for slot in range(max_slots):
        # start_id 传入 total_generated 以保证 ID 全局唯一且连续
        reqs = generator.generate_requests_poisson(
            lam=lam, 
            start_id=total_generated
        )
        request_dataset[slot] = reqs
        total_generated += len(reqs)

    # 生成完毕后恢复系统的随机混沌状态，防污染
    random.seed(None)
    np.random.seed(None)
    print(f"  - Dataset Ready: {total_generated} requests total.")
    print(f"[{time.strftime('%H:%M:%S')}] >>> Environment Setup Complete.\n")
    
    return base_network, request_dataset

# ==========================================
# 2. 实验执行器
# ==========================================
def run_experiment(name, metric_class, base_network, request_dataset):
    """
    运行单次实验
    Args:
        name: 实验名称 (用于打印)
        metric_class: 路由指标类 (UECMetric 或 MinHopsMetric)
        base_network: 基准网络 (将被深拷贝)
        request_dataset: 基准请求 (将被注入)
    """
    print("="*60)
    print(f"STARTING SIMULATION: {name}")
    print("="*60)
    
    
    # 1. 初始化仿真器     
    sim = QuantumSimulator(metric_class, external_network=base_network)
    
    # 手动同步配置 (确保 Simulator 使用我们要的参数)
    sim.max_slots = SIMULATION_CONFIG["MAX_SLOTS"]
    sim.requests_per_slot = SIMULATION_CONFIG["REQUESTS_PER_SLOT"]
    
    # 2. 开启预加载模式
    sim.set_preloaded_requests(request_dataset)
    
    # 3. 运行仿真
    sim.run()

    print(f"\n>>> Simulation {name} Finished.")
    return sim

# ==========================================
# 3. 结果分析与对比
# ==========================================

def print_comparison(sim_minhops, sim_cr, sim_uec):
    """
    打印三方对比表格 (MinHops vs CR vs UEC)
    以 MinHops 为基准，计算 CR 和 UEC 的提升率
    """
    def get_metrics(sim):
        # 1. 吞吐量 (Throughput)
        duration = sim.current_slot + 1
        tp = sim.total_allocated_bits / duration if duration > 0 else 0
        
        # 2. 资源效率 (Resource Efficiency)
        eff = sim.total_allocated_bits / sim.total_consumed_pairs if sim.total_consumed_pairs > 0 else 0
        
        # 3. 成功率 (Success Rate)
        sr = len(sim.completed_requests) / sim.total_requests if sim.total_requests > 0 else 0
        
        return tp, eff, sr

    # 1. 获取所有数据
    tp_base, eff_base, sr_base = get_metrics(sim_minhops)
    tp_cr,   eff_cr,   sr_cr   = get_metrics(sim_cr)
    tp_uec,  eff_uec,  sr_uec  = get_metrics(sim_uec)

    # 2. 辅助函数：格式化 "数值 (提升率)"
    def fmt_cell(val, base_val, is_rate=False):
        """
        val: 当前值
        base_val: 基准值 (MinHops)
        is_rate: 是否为百分比数据 (成功率)，如果是，则计算 pp 差异
        """
        # A. 格式化数值部分
        if is_rate:
            val_str = f"{val:.2%}"
        else:
            val_str = f"{val:.4f}"

        # B. 计算提升部分
        if is_rate:
            # 成功率用百分点 (pp)
            diff = (val - base_val) * 100
            if diff > 0: sign = "+"
            else: sign = "" # 负号会自动带
            imp_str = f"{sign}{diff:.2f} pp"
        else:
            # 吞吐量和效率用百分比 (%)
            if base_val > 0:
                imp = (val - base_val) / base_val * 100
            else:
                imp = 0.0
            imp_str = f"{imp:+.2f}%"

        return f"{val_str} ({imp_str})"

    # 3. 打印表头
    print("\n\n")
    print("#"*85)
    print(f"{'FINAL PERFORMANCE COMPARISON (Baseline: MinHops)':^85}")
    print("#"*85)
    
    # 调整列宽：Metric | MinHops | CR (Imp) | UEC (Imp)
    # 格式: Metric名 (22) | 基准值 (18) | CR值+提升 (22) | UEC值+提升 (22)
    headers = ["Metric", "MinHops (Base)", "CR (vs Base)", "UEC (vs Base)"]
    row_format = "{:<22} | {:<18} | {:<22} | {:<22}"
    
    print(row_format.format(*headers))
    print("-" * 90)
    
    # 4. 打印数据行
    # 为了表格整洁，我们将单位移到 Metric 名称中，不再塞入数据单元格
    
    # --- Throughput ---
    print(row_format.format(
        "Throughput (bits/slot)", 
        f"{tp_base:.4f}", 
        fmt_cell(tp_cr, tp_base),
        fmt_cell(tp_uec, tp_base)
    ))
    
    # --- Efficiency ---
    print(row_format.format(
        "Efficiency (bits/pair)", 
        f"{eff_base:.4f}", 
        fmt_cell(eff_cr, eff_base),
        fmt_cell(eff_uec, eff_base)
    ))
    
    # --- Success Rate ---
    print(row_format.format(
        "Success Rate", 
        f"{sr_base:.2%}", 
        fmt_cell(sr_cr, sr_base, is_rate=True),
        fmt_cell(sr_uec, sr_base, is_rate=True)
    ))
    
    print("-" * 90)
    print(f"Total Requests: {sim_minhops.total_requests} (Same for all)")
    print("#"*85)

# ==========================================
# Main 入口
# ==========================================
if __name__ == "__main__":
    # 准备环境
    fixed_network, fixed_requests = generate_benchmark_environment()

    sim_minhops = run_experiment("MinHops", MinHopsMetric, fixed_network, fixed_requests)
    sim_cr = run_experiment("CR", CRMetric, fixed_network, fixed_requests)
    sim_uec = run_experiment("UEC-Routing", UECMetric, fixed_network, fixed_requests)
    
    # 结果对比
    print_comparison(sim_minhops, sim_cr, sim_uec)