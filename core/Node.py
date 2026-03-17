"""
    量子节点
"""
from dataclasses import dataclass

@dataclass
class QuantumNode:
    node_id: int
    x_coord: float
    y_coord: float
    
    # 资源属性 
    max_rate: int      # Re
    max_memory: int      # M
    
    # 节点占用状态
    current_rate: int = 0
    current_memory: int = 0

    # 修改动态状态的方法
    def reset(self):
        """
        重置节点占用状态
        """
        self.current_rate = 0
        self.current_memory = 0

    def consume(self, re_demand, mem_demand):
        """修改节点占用状态"""
        self.current_rate += re_demand
        self.current_memory += mem_demand
