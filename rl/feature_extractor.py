"""
Quantum_RL/rl/feature_extractor.py
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class QuantumRoutingExtractor(BaseFeaturesExtractor):
    """
    针对 QuantumRLEnv 字典观测空间的自定义特征提取器 (V10 上帝视角专版)
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # 初始化基类 (V10 默认输出 512 维的高阶特征)
        super().__init__(observation_space, features_dim)

        # ==========================================
        # 1. 资源特征提取 [N, 2] -> 展平为 [N*2]
        # ==========================================
        res_shape = observation_space.spaces["network_resources"].shape
        num_nodes = res_shape[0]
        # 扩大隐藏层，让它更好地理解整张网的宏观瓶颈
        self.res_extractor = nn.Sequential(
            nn.Linear(num_nodes * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # ==========================================
        # 2. [W, K, max_hops, 4] -> 展平为 [W * K * max_hops * 4]
        # ==========================================
        path_shape = observation_space.spaces["request_paths"].shape
        W_val = path_shape[0]
        K_val = path_shape[1]
        max_hops = path_shape[2]

        path_input_dim = W_val * K_val * max_hops * 4  # 例如 10 * 5 * 6 * 4 = 1200 维

        self.path_extractor = nn.Sequential(
            nn.Linear(path_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # ==========================================
        # 3. 需求量与上下文特征融合
        # demand: [W]维, context: [2]维 -> 拼接成 [W + 2] 维
        # ==========================================
        demand_shape = observation_space.spaces["request_demand"].shape
        ctx_shape = observation_space.spaces["global_context"].shape

        ctx_input_dim = demand_shape[0] + ctx_shape[0]  # 例如 10 + 2 = 12 维

        self.context_extractor = nn.Sequential(
            nn.Linear(ctx_input_dim, 64),
            nn.ReLU()
        )

        # ==========================================
        # 4. 融合层
        # 将三大模块提纯后的结果拼在一起：128(资源) + 256(路径) + 64(上下文) = 448
        # ==========================================
        concat_dim = 128 + 256 + 64
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播：PPO 将每步的状态字典传进来，在这里被炼化为最终的向量
        """
        # 1. 压扁并提取资源特征
        res_obs = observations["network_resources"].float()  # [batch_size, N, 2]
        res_flat = torch.flatten(res_obs, start_dim=1)  # [batch_size, N*2]
        res_feat = self.res_extractor(res_flat)  # [batch_size, 128]

        # 2. 压扁并提取路径特征 (这里接住了 1200 维的庞大矩阵)
        path_obs = observations["request_paths"].float()
        path_flat = torch.flatten(path_obs, start_dim=1)
        path_feat = self.path_extractor(path_flat)  # [batch_size, 256]

        # 3. 将 [W] 维的需求和 [2] 维的上下文拼接，提取高级意图
        demand_obs = observations["request_demand"].float()  # [batch_size, W]
        ctx_obs = observations["global_context"].float()  # [batch_size, 2]
        combined_ctx = torch.cat([demand_obs, ctx_obs], dim=1)  # [batch_size, W + 2]
        ctx_feat = self.context_extractor(combined_ctx)  # [batch_size, 64]

        # 4. 提纯出最终的 features_dim (512 维) 传给 PPO 主网络！
        fused_features = torch.cat([res_feat, path_feat, ctx_feat], dim=1)  # [batch_size, 448]
        final_features = self.fusion_net(fused_features)  # [batch_size, 512]

        return final_features