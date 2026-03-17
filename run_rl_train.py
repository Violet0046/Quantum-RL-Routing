"""
run_rl_train.py
强化学习模型训练入口
"""
from scheduler.RLScheduler import RLScheduler

if __name__ == "__main__":

    print("=" * 50)
    print("🚀 启动 Quantum RL 训练引擎...")
    print("=" * 50)

    # 1. 实例化统管中心 (自动读取 rl/hyperparams.yml 配置)
    scheduler = RLScheduler()

    # 2. 一键启动训练 (包含无痕建网、PPO 初始化、回调监控和模型保存)cl
    scheduler.train()

    print("🎉 训练引擎运行结束！")