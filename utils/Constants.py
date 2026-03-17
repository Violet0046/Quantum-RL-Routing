"""
常量配置
"""
# 物理/环境常数
ALPHA_FIBER = 0.046        # 光纤衰减系数 (0.2dB/km)
DEPOLAR_COEFF = 0.005     # 去极化系数 (1/km)   影响链路保真度，会导致提纯次数增加
F_INIT = 0.99             # 节点制备纠缠的初始保真度上限
P_SWAP_DEFAULT = 0.5      # 纠缠交换成功率

# KSP配置
KSP_CONFIG = {
    "K": 5,    #   KSP的K值
    "FIDELITY_WEIGHT" : 1.0, # 影响边权重   total_weight = weight_connectivity + KSP_CONFIG.get("FIDELITY_WEIGHT", 1.0) * weight_fidelity
}                            # 边权重 = 基本都是weight_connectivity占主导

# 资源配置模板 
RES_CONFIG = {
    "HIGH": {"rate": 500.0, "memory": 100},   # 骨干节点   存储器数量等价为路径的宽度
    "LOW":  {"rate": 100.0,  "memory": 20}    # 边缘节点
}

# 拓扑与环境配置
TOPO_CONFIG = {
    "WIDTH": 100.0,     # 区域宽度 (km)
    "HEIGHT": 100.0,    # 区域高度 (km)
    "NUM_NODES": 50,    # 节点数  50 100 200 400 800
    "ALPHA": 0.4,       # 决定网络中链路生成的基准概率幅值
    "BETA": 0.4,        # 影响长距离链路的生成概率
}

# 请求范围配置
REQUEST_CONFIG = {
    "MIN_NUM_PAIRS": 1,
    "MAX_NUM_PAIRS": 5,
    # 标准服务等级 (Service Classes),列表中的值代表标准保真度
    "FIDELITY_LEVELS": [0.7, 0.75, 0.8, 0.85],
    # 40%是低端需求，10%是高端需求
    "FIDELITY_PROBABILITIES": [0.5, 0.3, 0.15, 0.05]
}

# 调度器配置
SCHEDULER_CONFIG = {
    "ALPHA": 0.0,       # sort_score = cost - (ALPHA * waiting_slots) 优先为sort_score小的请求路径分配资源
}

# 仿真配置
SIMULATION_CONFIG = {
    "MAX_WAIT": 50,            # 超时请求阈值
    "REQUESTS_PER_SLOT": 5,    # 每时隙生成的新请求数
    "LAM": 5,                  # 泊松分布的期望值 Lambda (即平均每时隙产生的请求数)
    "MAX_SLOTS": 1000      # 最大仿真时隙数
}


