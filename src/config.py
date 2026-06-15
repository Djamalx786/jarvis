"""Central, dated life constraints for the Life OS Agent.

These used to live as relative phrases ("in 3 Wochen") inside the RAG notes, which
silently went stale as time passed. Pinning them to absolute dates here makes the
planner correct regardless of when it runs.
"""
from datetime import date

# Djamal starts full-time at BMW on this date. Before it, weekday daytimes are FREE
# (no fixed work/commute blocks); on/after it the fixed BMW work day applies.
BMW_START_DATE = date(2026, 7, 1)

# Strength training is medically off-limits until this date (foot/health break).
# Before it: only low-impact movement (walking, easy cycling, stretching, foot exercises).
STRENGTH_TRAINING_ALLOWED_FROM = date(2026, 8, 1)
