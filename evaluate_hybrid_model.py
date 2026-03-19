"""
evaluate_hybrid_model.py
"""
# 1. 导入混合调度器
from scheduler.HybridScheduler import HybridScheduler
from utils.generate_benchmark_environment import generate_benchmark_environment

if __name__ == "__main__":
    base_network, request_dataset = generate_benchmark_environment()

    # 2. 实例化混合调度中心
    scheduler = HybridScheduler()

    MODEL_PATH = "experiments/models/ppo_hybrid_V1.zip"

    print(">>> 正在启动 RL + Greedy 两阶段混合评估...")
    scheduler.evaluate(model_path=MODEL_PATH, test_pool=request_dataset)