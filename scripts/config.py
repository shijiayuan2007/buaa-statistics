"""Project-level configuration for the Iran-Israel oil price event study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"

START_DATE = "2024-01-01"
END_DATE = "2026-05-20"

ESTIMATION_WINDOW = 120
EVENT_WINDOW_PRE = 5
EVENT_WINDOW_POST = 10
ITS_WINDOW_DAYS = 60
SIGNIFICANCE_LEVEL = 0.05

# 7 key events, selected from the assignment narrative and the existing project.
EVENTS = [
    {
        "event_id": "E1",
        "date": "2024-10-01",
        "name": "伊朗导弹袭击以色列",
        "description": "伊朗向以色列发射约200枚弹道导弹",
        "type": "escalation",
    },
    {
        "event_id": "E2",
        "date": "2024-10-26",
        "name": "以色列报复性打击伊朗",
        "description": "以色列对伊朗军事目标实施报复性空袭",
        "type": "escalation",
    },
    {
        "event_id": "E3",
        "date": "2025-06-13",
        "name": "十二日战争爆发",
        "description": "以色列空袭伊朗核设施和军事目标",
        "type": "escalation",
    },
    {
        "event_id": "E4",
        "date": "2025-06-24",
        "name": "十二日战争停火",
        "description": "美国斡旋下伊以达成停火协议",
        "type": "de-escalation",
    },
    {
        "event_id": "E5",
        "date": "2026-02-28",
        "name": "美以联合打击伊朗",
        "description": "美国和以色列联合对伊朗发动打击，全面战争爆发",
        "type": "escalation",
    },
    {
        "event_id": "E6",
        "date": "2026-03-11",
        "name": "伊朗封锁霍尔木兹海峡",
        "description": "伊朗攻击海峡油轮，实质性封锁霍尔木兹海峡",
        "type": "escalation",
    },
    {
        "event_id": "E7",
        "date": "2026-04-30",
        "name": "油价触顶与停火谈判",
        "description": "Brent触及高点，停火谈判启动",
        "type": "de-escalation",
    },
]
