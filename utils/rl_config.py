# utils/rl_config.py
class RLConfig:
    # 环境限制
    MAX_SLOTS_TRAIN = 400   # 训练最长时隙
    MAX_SLOTS_EVAL = 1000   # 评估时隙
    MAX_STEPS_PER_EPISODE = 15000   # 每Episode最大步数

    W = 10    # 前瞻窗口大小
