"""mm_event_agent 命令行 / 服务入口。"""

from __future__ import annotations

from mm_event_agent.config import settings
from mm_event_agent.graph import build_graph


def main() -> None:
    if settings.debug:
        pass  # 可在此打开详细日志
    build_graph()


if __name__ == "__main__":
    main()
