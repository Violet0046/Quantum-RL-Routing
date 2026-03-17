"""
    量子链路
"""
from dataclasses import dataclass, field
import math
from utils.Constants import ALPHA_FIBER, DEPOLAR_COEFF, F_INIT

@dataclass
class QuantumLink:
    u: int
    v: int
    length: float  # km

    # 在 __post_init__ 中计算，保证物理一致性
    attenuation: float = field(init=False)  # Transmittance (eta)
    fidelity: float = field(init=False)     # F_phys

    def __post_init__(self):
        # 1. 链路衰减因子
        self.attenuation = math.exp(-ALPHA_FIBER * self.length)

        # 2. 链路综合保真度
        self.fidelity = (1 + 3 * F_INIT  * math.exp(-DEPOLAR_COEFF * self.length)) / 4