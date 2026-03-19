"""
run_hybrid_train.py
"""
from scheduler.HybridScheduler import HybridScheduler

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 启动 RL + Greedy 混合大脑训练引擎...")
    print("=" * 50)

    # 实例化混合调度器
    scheduler = HybridScheduler()

    scheduler.exp_name = "ppo_hybrid_v1"

    scheduler.train()

    print("🎉 训练引擎运行结束！")