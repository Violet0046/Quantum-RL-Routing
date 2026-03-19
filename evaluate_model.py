"""
evaluate_model.py
强化学习模型推理与测试入口
"""
from scheduler.RLScheduler import RLScheduler
from utils.generate_benchmark_environment import generate_benchmark_environment
from routing.metrics.MinHops import MinHopsMetric
from routing.metrics.CR import CRMetric
from routing.metrics.UEC import UECMetric
from run_base_comparison import run_experiment

if __name__ == "__main__":

    base_network, request_dataset = generate_benchmark_environment()

# ----------------------进行基线测试 内部均为深拷贝----------------------
    # sim_min_hops = run_experiment("MinHops", MinHopsMetric, base_network, request_dataset)
    # sim_cr = run_experiment("CR", CRMetric, base_network, request_dataset)
    # sim_uec = run_experiment("UEC-Routing", UECMetric, base_network, request_dataset)

# ----------------------     进行RL模型测试     ----------------------
    # 1. 实例化统管中心
    scheduler = RLScheduler()   #内部用固定自行生成拓扑，传入环境

    # 2. 指定测试的模型路径
    MODEL_PATH = "experiments/models/ppo_hybrid_V1.zip"

    # 3. 启动评估
    scheduler.evaluate(model_path = MODEL_PATH, test_pool = request_dataset)   #深拷贝请求