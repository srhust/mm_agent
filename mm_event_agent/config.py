"""应用与运行环境配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """集中存放可调参数，后续可从环境变量加载。"""

    debug: bool = os.getenv("MM_EVENT_DEBUG", "").lower() in ("1", "true", "yes")


settings = Settings()
