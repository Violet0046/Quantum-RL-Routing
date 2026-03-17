"""
Quantum_RL/rl/callbacks.py
"""
from stable_baselines3.common.callbacks import BaseCallback

class QuantumMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                # 去 final_info 里获取被 SB3 自动重置隐藏起来的成绩单
                if 'final_info' in info and 'uec_results' in info['final_info']:
                    ue = info['final_info']['uec_results']
                    self.episode_count += 1

                    # 记录到 TensorBoard
                    self.logger.record("uec_results/success_rate", ue['success_rate'])
                    self.logger.record("uec_results/throughput", ue['throughput'])
                    self.logger.record("uec_results/efficiency", ue['efficiency'])

                    # 打印汇报
                    if self.episode_count % 2 == 0:
                        print(f"\n🏆 [Episode {self.episode_count} 结束] 成功率: {ue['success_rate']:.2%} | 吞吐量: {ue['throughput']:.2f} | 资源效率: {ue['efficiency']:.4f}")
        return True