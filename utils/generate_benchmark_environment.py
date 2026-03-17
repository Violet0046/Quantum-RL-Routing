import random
import numpy as np
from core.Network import QuantumNetworkManager
from utils.RequestGenerator import RequestGenerator
from utils.Constants import SIMULATION_CONFIG

def generate_benchmark_environment():
    """
    生成:
    1. 固定的物理拓扑
    2. 固定的请求数据集 (覆盖所有时隙)
    """
    TOPO_SEED = 202433851
    TEST_REQ_SEED = 8888
    # 1. 生成基准拓扑
    random.seed(TOPO_SEED)
    np.random.seed(TOPO_SEED)
    base_network = QuantumNetworkManager()
    base_network.generate_waxman_topology()
    # 恢复系统的随机混沌状态，防污染
    random.seed(None)
    np.random.seed(None)

    # 2. 生成基准请求集
    random.seed(TEST_REQ_SEED)
    np.random.seed(TEST_REQ_SEED)
    generator = RequestGenerator(base_network)
    request_dataset = {}
    total_generated = 0
    max_slots = SIMULATION_CONFIG.get('MAX_SLOTS')
    lam = SIMULATION_CONFIG.get('LAM')

    for slot in range(max_slots):
        # start_id 传入 total_generated 以保证 ID 全局唯一且连续
        reqs = generator.generate_requests_poisson(
            lam=lam,
            start_id=total_generated
        )
        request_dataset[slot] = reqs
        total_generated += len(reqs)
    # 恢复系统的随机混沌状态，防污染
    random.seed(None)
    np.random.seed(None)
    return base_network, request_dataset