"""
Quantum_RL/scheduler/RLScheduler.py
强化学习统管中心：负责组装环境、统筹训练与测试
"""
import os
import time
import yaml
import random
import numpy as np
import csv
import copy
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from rl.callbacks import QuantumMetricsCallback
from rl.feature_extractor import QuantumRoutingExtractor
from stable_baselines3.common.monitor import Monitor

from core.Network import QuantumNetworkManager
from routing.KSP import QuantumRouter
from routing.metrics.UEC import UECMetric
from utils.RequestGenerator import RequestGenerator
from utils.linear_schedule import linear_schedule
from rl.env import QuantumRLEnv
from utils.rl_config import RLConfig

class RLScheduler:
    def __init__(self, config_path: str = "rl/hyperparams.yml"):
        """
        初始化 RL 调度器，读取配置并准备好统一的基础设施
        """
        # 1. 读取超参数
        self.config = self._load_config(config_path)
        self.exp_name = self.config.get("experiment_name", "ppo_uec_run")

        # 提取种子配置
        self.topo_seed = self.config["seed"].get("topo_seed", 202433851)

        self.metric = UECMetric()

    @staticmethod
    def _load_config(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_env(self, max_slots=None, max_steps=None) -> QuantumRLEnv:
        """
        装配并返回一个全新的 QuantumRLEnv 环境
        """
        # 1. 固定拓扑种子，确保拓扑结构永远一致
        random.seed(self.topo_seed)
        np.random.seed(self.topo_seed)

        network = QuantumNetworkManager()
        network.generate_waxman_topology()
        # 传入 None，重置随机种子
        random.seed(None)
        np.random.seed(None)
        # 2. 组装路由与生成器
        router = QuantumRouter(network.graph, self.metric)
        generator = RequestGenerator(network)
        slots = max_slots if max_slots is not None else RLConfig.MAX_SLOTS_TRAIN
        steps = max_steps if max_steps is not None else RLConfig.MAX_STEPS_PER_EPISODE
        # 3. 实例化环境并返回
        env = QuantumRLEnv(network, router, generator, slots, steps)
        return env

    def train(self):
        """
        训练主流程
        """
        # 1. 生成固定拓扑网络
        env = self.build_env(max_slots=RLConfig.MAX_SLOTS_TRAIN,
            max_steps=RLConfig.MAX_STEPS_PER_EPISODE)
        train_env = Monitor(env)

        # 2. 配置并初始化 PPO 智能体
        policy_kwargs = dict(
            features_extractor_class=QuantumRoutingExtractor,
            features_extractor_kwargs=dict(features_dim=self.config["policy"]["features_dim"]),
            net_arch=self.config["policy"]["net_arch"]
        )
        # 生成动态的时间戳目录名
        os.makedirs("experiments/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.exp_name}_{timestamp}"
        tensorboard_root = f"./experiments/logs"

        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=linear_schedule(self.config["ppo"]["learning_rate"]),
            #learning_rate=self.config["ppo"]["learning_rate"],
            n_steps=self.config["ppo"]["n_steps"],
            batch_size=self.config["ppo"]["batch_size"],
            gamma=self.config["ppo"]["gamma"],
            ent_coef=self.config["ppo"]["ent_coef"],
            clip_range=self.config["ppo"]["clip_range"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_root,
            verbose=1
        )
        # 添加这两行：
        print(f"🔥 核心检查：模型当前运行在设备 -> {model.device}")
        print(f"显卡型号: {torch.cuda.get_device_name(model.device) if 'cuda' in str(model.device) else 'None'}")

        # 4. 开始训练
        metrics_callback = QuantumMetricsCallback()
        total_steps = self.config["training"]["total_steps"]
        print(f">>> 开始训练，总步数: {total_steps}")
        model.learn(total_timesteps=total_steps, callback=metrics_callback)

        # 5. 保存模型
        os.makedirs("experiments/models", exist_ok=True)
        save_path = f"experiments/models/{run_id}.zip"
        model.save(save_path)
        print(f"\n>>> 模型已保存至 {save_path}")

    def evaluate(self, model_path: str, test_pool):
        """
        评估已训练模型
        """
        # 1. 生成固定拓扑网络
        env = self.build_env(max_slots=RLConfig.MAX_SLOTS_EVAL,
            max_steps=float('inf'))

        # 2. 打开预加载模式
        env.use_preloaded = True
        env.preloaded_pool = copy.deepcopy(test_pool)

        # 3. 加载模型与 CSV 记录器
        model = PPO.load(model_path, device="cpu")
        csv_filename = self._prepare_csv_logger()

        # 4. 重置环境
        obs, _ = env.reset()
        done = False
        current_displayed_slot = 0
        pbar = tqdm(total=env.max_slots, desc="RL Inference", dynamic_ncols=True, unit="slot")

        # 5. 开始推演 (在循环外统一打开文件)
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)  # 实例化一次 writer

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 更新进度条
                if env.current_slot > current_displayed_slot:
                    pbar.update(env.current_slot - current_displayed_slot)
                    current_displayed_slot = env.current_slot

                # 提取数据直接写入，无需反复打开文件
                if 'snapshot' in info:
                    csv_writer.writerow(info['snapshot'])

        pbar.close()
        print(f"\n>>> 评估报告已保存至 {csv_filename}")

    def _prepare_csv_logger(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_filename = f"experiments/results/rl_eval_{self.exp_name}_{timestamp}.csv"
        os.makedirs("experiments/results", exist_ok=True)
        headers = ["Slot", "Total_Requests", "Completed", "Pending", "Blocked", "Success_Rate", "Throughput_Avg",
                   "Efficiency_Avg"]
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
        return csv_filename