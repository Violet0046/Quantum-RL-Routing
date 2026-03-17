from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率衰减策略。
    :param initial_value: 初始学习率 (例如 3e-4)
    :return: 一个接收剩余进度 progress_remaining (1.0 -> 0.0) 并返回当前学习率的函数
    """
    def func(progress_remaining: float) -> float:
        # 随着训练进行，progress_remaining 会从 1.0 匀速下降到 0.0
        return progress_remaining * initial_value
    return func