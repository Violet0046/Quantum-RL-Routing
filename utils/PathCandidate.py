from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PathCandidate:
    """
    路径候选者包装类:将请求的K条路径进行包装,调度池中使用sort_score进行排序
    """
    # 排序依据 (Score)，放在第一位，Python 默认基于此字段排序
    sort_score: float = field(compare=True)

    request: Any = field(compare=False)
    path_index: int = field(compare=False)
