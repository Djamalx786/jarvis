"""Weekly plan PDF export.

Renders the structured weekly plan as a proper *time-grid* calendar (hours on the
vertical axis, the seven days on the horizontal axis) on a single landscape A4 page.
Each task becomes a colored block positioned by its start time and sized by its
duration — so the page reads like a real planner instead of a sparse list.
"""
from datetime import date, datetime, timedelta

from weasyprint import HTML

# Category -> (pastel background, accent / text color)
CATEGORY_COLORS = {
    "Arbeit (BMW)":      ("#C7D2FE", "#1E3A8A"),
    "KI & Projekte":     ("#EDE9FE", "#6D28D9"),
    "Islam & Reflexion": ("#DCFCE7", "#15803D"),
    "Familie & Ehe":     ("#FFE4E6", "#BE123C"),
    "Sport & Freizeit":  ("#FEF3C7", "#B45309"),
}
DEFAULT_STYLE = ("#F8FAFC", "#334155")

# Legend order + short labels shown in the header.
LEGEND = [
    ("Arbeit (BMW)", "Arbeit"),
    ("KI & Projekte", "KI & Projekte"),
    ("Islam & Reflexion", "Islam"),
    ("Familie & Ehe", "Familie"),
    ("Sport & Freizeit", "Sport & Freizeit"),
]

# Category -> Google Calendar event colorId (1-11).
# https://developers.google.com/calendar/api/v3/reference/colors
CATEGORY_GCAL_COLORS = {
    "Arbeit (BMW)":      "9",   # Blueberry - dunkelblau
    "KI & Projekte":     "3",   # Grape - lila
    "Islam & Reflexion": "10",  # Basil - grün
    "Familie & Ehe":     "4",   # Flamingo - rosa/rot
    "Sport & Freizeit":  "5",   # Banana - gelb/orange
}

DAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
DAY_ABBR = ["MO", "DI", "MI", "DO", "FR", "SA", "SO"]

# Default visible time window; expands automatically if tasks fall outside it.
DEFAULT_START_HOUR = 7
DEFAULT_END_HOUR = 22

# Physical height (mm) of the time grid area on the page. Event positions are
# computed as percentages of the visible time span within this height.
GRID_HEIGHT_MM = 150


def get_chip_style(category: str):
    return CATEGORY_COLORS.get(category, DEFAULT_STYLE)


def _parse_time_range(time_str: str):
    """Parse 'HH:MM - HH:MM' into (start_minutes, end_minutes). Returns None if invalid."""
    if not time_str:
        return None
    cleaned = time_str.replace("–", "-").replace("—", "-")
    parts = [p.strip() for p in cleaned.split("-")]
    if len(parts) != 2:
        return None
    try:
        sh, sm = (int(x) for x in parts[0].split(":"))
        eh, em = (int(x) for x in parts[1].split(":"))
    except (ValueError, IndexError):
        return None
    start = sh * 60 + sm
    end = eh * 60 + em
    if end <= start:
        end = start + 60
    return start, end


def _collect_timed_tasks(plan_data: dict):
    """Return {day_index: [task_with_minutes]} and the (start_hour, end_hour) window to render."""
    by_day = {}
    min_start = DEFAULT_START_HOUR * 60
    max_end = DEFAULT_END_HOUR * 60

    for i, day in enumerate(DAYS):
        tasks = []
        for task in plan_data.get(day, []):
            parsed = _parse_time_range(task.get("time", ""))
            if parsed is None:
                continue
            start, end = parsed
            min_start = min(min_start, start)
            max_end = max(max_end, end)
            tasks.append({
                "start": start,
                "end": end,
                "title": task.get("title", "").strip(),
                "category": task.get("category", ""),
            })
        by_day[i] = tasks

    start_hour = min_start // 60
    end_hour = -(-max_end // 60)  # ceil
    return by_day, start_hour, end_hour


def _assign_lanes(tasks: list):
    """Assign overlapping tasks to side-by-side lanes. Mutates tasks with 'lane'/'lanes'."""
    if not tasks:
        return
    tasks.sort(key=lambda t: (t["start"], t["end"]))

    # Split into clusters of chain-overlapping tasks.
    clusters, current, cluster_end = [], [], None
    for t in tasks:
        if current and t["start"] >= cluster_end:
            clusters.append(current)
            current, cluster_end = [], None
        current.append(t)
        cluster_end = t["end"] if cluster_end is None else max(cluster_end, t["end"])
    if current:
        clusters.append(current)

    for cluster in clusters:
        lane_ends = []
        for t in cluster:
            placed = False
            for idx, end in enumerate(lane_ends):
                if t["start"] >= end:
                    t["lane"], lane_ends[idx], placed = idx, t["end"], True
                    break
            if not placed:
                t["lane"] = len(lane_ends)
                lane_ends.append(t["end"])
        for t in cluster:
            t["lanes"] = len(lane_ends)


def _fmt_minutes(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _build_event_blocks(tasks: list, span_start_min: int, span_total_min: int) -> str:
    _assign_lanes(tasks)
    html = ""
    for t in tasks:
        bg, fg = get_chip_style(t["category"])
        top = (t["start"] - span_start_min) / span_total_min * 100
        height = (t["end"] - t["start"]) / span_total_min * 100
        lanes = t.get("lanes", 1)
        lane = t.get("lane", 0)
        width = 100 / lanes
        left = lane * width
        gap = 1.5 if lanes > 1 else 0
        time_label = f"{_fmt_minutes(t['start'])}–{_fmt_minutes(t['end'])}"
        # Compact layout for short blocks: keep time + title on one line.
        compact = "compact" if height < 6 else ""
        html += (
            f'<div class="event {compact}" style="top:{top:.2f}%;height:{height:.2f}%;'
            f'left:calc({left:.2f}% + {gap}px);width:calc({width:.2f}% - {gap * 2}px);'
            f'background:{bg};border-left-color:{fg};color:{fg};">'
            f'<span class="event-time">{time_label}</span>'
            f'<span class="event-title">{t["title"]}</span>'
            f'</div>'
        )
    return html


def generate_pdf(plan_data: dict, week_start: date | None = None) -> str:
    """Render a weekly plan dict to a one-page landscape A4 time-grid PDF.

    plan_data maps German day names (Montag..Sonntag) to lists of
    {"time": "HH:MM - HH:MM", "title": str, "category": str} entries.
    """
    if week_start is None:
        today = datetime.now().date()
        week_start = today + timedelta(days=(7 - today.weekday()))
    elif isinstance(week_start, datetime):
        week_start = week_start.date()

    week = week_start.strftime("%Y-%W")
    output_path = f"data/exports/weekplan_{week}.pdf"

    dates = [week_start + timedelta(days=i) for i in range(7)]
    today = datetime.now().date()

    by_day, start_hour, end_hour = _collect_timed_tasks(plan_data)
    span_start_min = start_hour * 60
    hours_visible = end_hour - start_hour
    span_total_min = hours_visible * 60
    hour_mm = GRID_HEIGHT_MM / hours_visible

    total_blocks = sum(len(v) for v in by_day.values())

    # ── Hour labels in the gutter (lines are drawn per column via gradient) ──
    hour_labels = ""
    for hour in range(start_hour, end_hour + 1):
        top = (hour - start_hour) / hours_visible * 100
        hour_labels += f'<div class="hour-label" style="top:{top:.3f}%;">{hour:02d}</div>'

    # ── Day header cells ──
    day_heads = '<div class="head-cell gutter-head"></div>'
    for i in range(7):
        is_today = dates[i] == today
        is_weekend = i >= 5
        cls = "head-cell"
        if is_today:
            cls += " is-today"
        elif is_weekend:
            cls += " is-weekend"
        day_heads += (
            f'<div class="{cls}">'
            f'<span class="head-abbr">{DAY_ABBR[i]}</span>'
            f'<span class="head-date">{dates[i].day}</span>'
            f'</div>'
        )

    # ── Day columns with positioned event blocks ──
    day_cols = ""
    for i in range(7):
        is_today = dates[i] == today
        is_weekend = i >= 5
        cls = "day-col"
        if is_today:
            cls += " is-today"
        elif is_weekend:
            cls += " is-weekend"
        blocks = _build_event_blocks(by_day[i], span_start_min, span_total_min)
        day_cols += f'<div class="{cls}">{blocks}</div>'

    week_range = f"{dates[0].strftime('%d. %b')} – {dates[6].strftime('%d. %b %Y')}"
    kw = week_start.strftime("%W")

    legend_html = "".join(
        f'<div class="legend-item"><span class="legend-dot" style="background:{CATEGORY_COLORS[cat][1]};"></span>{label}</div>'
        for cat, label in LEGEND
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

@page {{ size: A4 landscape; margin: 9mm 10mm; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #0F172A;
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

/* ── Header ── */
.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 8px;
}}
.header-left {{ display: flex; align-items: center; gap: 11px; }}
.logo-mark {{
    width: 34px; height: 34px;
    background: #0F172A; color: white;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; letter-spacing: -0.5px;
}}
.title {{ font-size: 17px; font-weight: 800; letter-spacing: -0.4px; line-height: 1.1; }}
.title span {{ color: #2563EB; }}
.subtitle {{ font-size: 9px; color: #94A3B8; font-weight: 500; letter-spacing: 0.3px; margin-top: 1px; }}
.header-right {{ text-align: right; }}
.week-range {{ font-size: 13px; font-weight: 700; color: #0F172A; }}
.week-kw {{ font-size: 9px; color: #94A3B8; font-weight: 600; letter-spacing: 1px; margin-top: 2px; }}

/* ── Legend ── */
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; padding: 1px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 8.5px; font-weight: 600; color: #475569; white-space: nowrap; }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 3px; }}

/* ── Day header row ── */
.cal-head {{
    display: grid;
    grid-template-columns: 34px repeat(7, 1fr);
    gap: 4px;
}}
.head-cell {{
    display: flex; align-items: baseline; justify-content: center; gap: 5px;
    padding: 5px 0;
    border-radius: 7px 7px 0 0;
    background: #F8FAFC;
}}
.head-cell.is-weekend {{ background: #F1F5F9; }}
.head-cell.is-today {{ background: #0F172A; }}
.head-abbr {{ font-size: 9px; font-weight: 700; letter-spacing: 1.5px; color: #64748B; }}
.head-date {{ font-size: 15px; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; }}
.is-today .head-abbr, .is-today .head-date {{ color: white; }}
.gutter-head {{ background: transparent; }}

/* ── Time grid ── */
.cal-body {{
    height: {GRID_HEIGHT_MM}mm;
    display: grid;
    grid-template-columns: 34px repeat(7, 1fr);
    gap: 4px;
}}
.gutter {{ position: relative; }}
.hour-label {{
    position: absolute;
    right: 5px; width: 26px;
    transform: translateY(-50%);
    font-size: 8px; font-weight: 600; color: #CBD5E1;
    text-align: right;
}}
.day-col {{
    position: relative;
    background-color: #FCFDFE;
    background-image: repeating-linear-gradient(
        to bottom,
        #EEF2F7 0, #EEF2F7 1px,
        transparent 1px, transparent {hour_mm:.4f}mm
    );
    border-radius: 0 0 7px 7px;
    border-left: 1px solid #EEF2F7;
    border-right: 1px solid #EEF2F7;
}}
.day-col.is-weekend {{ background-color: #F6F8FB; }}
.day-col.is-today {{ background-color: #F5F8FF; box-shadow: inset 0 0 0 1.5px #BFD3FF; }}

.event {{
    position: absolute;
    border-left: 3px solid;
    border-radius: 5px;
    padding: 3px 5px;
    overflow: hidden;
    box-shadow: 0 1px 1.5px rgba(15,23,42,0.07);
}}
.event-time {{ display: block; font-size: 7px; font-weight: 700; opacity: 0.8; letter-spacing: 0.2px; }}
.event-title {{ display: block; font-size: 8.5px; font-weight: 600; line-height: 1.2; margin-top: 1px; }}
.event.compact {{ padding: 1.5px 5px; }}
.event.compact .event-time {{ display: inline; margin-right: 4px; }}
.event.compact .event-title {{ display: inline; margin-top: 0; font-size: 8px; }}

/* ── Footer ── */
.footer {{
    display: flex; justify-content: space-between; align-items: center;
    font-size: 7.5px; color: #CBD5E1; font-weight: 500;
    letter-spacing: 0.5px;
}}
</style>
</head>
<body>

<div class="header">
    <div class="header-left">
        <div class="logo-mark">LO</div>
        <div>
            <div class="title">Wochenplan <span>Djamal</span></div>
            <div class="subtitle">Life OS · Personal AI Planner</div>
        </div>
    </div>
    <div class="header-right">
        <div class="week-range">{week_range}</div>
        <div class="week-kw">KALENDERWOCHE {kw}</div>
    </div>
</div>

<div class="legend">{legend_html}</div>

<div class="cal-head">{day_heads}</div>

<div class="cal-body">
    <div class="gutter">{hour_labels}</div>
    {day_cols}
</div>

<div class="footer">
    <span>{total_blocks} geplante Blöcke</span>
    <span>Erstellt {datetime.now().strftime("%d.%m.%Y")} · Life OS Agent</span>
</div>

</body>
</html>"""

    HTML(string=html).write_pdf(output_path)
    return output_path
