"""
dashboard.py — Optional Streamlit Dashboard for Face Tracker
Run with: streamlit run dashboard.py

Shows:
  - Live unique visitor count
  - Recent events table
  - Latest face entry/exit images
"""

import streamlit as st  # type: ignore
import sqlite3
import os
import json
from pathlib import Path
import pandas as pd  # type: ignore

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

DB_PATH  = cfg.get("db_path", "database/face_tracker.db")
LOG_DIR  = cfg.get("log_dir", "logs")

st.set_page_config(
    page_title="Face Tracker Dashboard",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Intelligent Face Tracker Dashboard")
st.caption("Katomaran Hackathon – March 2026 | Real-time Visitor Analytics")

import logging
logger = logging.getLogger(__name__)

# ── Helper ──
def get_db():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)


def unique_visitor_count() -> int:
    conn = get_db()
    if not conn:
        return 0
    try:
        row = conn.execute("SELECT COUNT(*) FROM faces").fetchone()
        return int(row[0]) if row else 0
    except Exception as e:
        logger.error(f"Error getting visitor count: {e}")
        return 0
    finally:
        conn.close()
    return 0


def recent_events(limit: int = 50) -> pd.DataFrame:
    conn = get_db()
    if not conn:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(
            f"SELECT face_id, event_type, timestamp, image_path "
            f"FROM events ORDER BY timestamp DESC LIMIT {limit}",
            conn,
        )
        return df
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    return pd.DataFrame()


def latest_images(event_type: str, count: int = 6) -> list[str]:
    """Return paths to the latest saved images for a given event type."""
    base = os.path.join(LOG_DIR, event_type + "s")
    images = sorted(
        Path(base).glob("**/*.jpg"),
        key=os.path.getmtime,
        reverse=True,
    )
    return [str(p) for p in images[:count] if p.exists()]  # type: ignore


# ── Layout ────────────────────────────────────────────────────────────────────
# Auto-refresh every 5 seconds
st_autorefresh = st.empty()

col1, col2, col3 = st.columns(3)
count = unique_visitor_count()
col1.metric("👤 Unique Visitors", count)
col2.metric("📂 DB Path", DB_PATH)
col3.metric("🖼️ Log Directory", LOG_DIR)

st.divider()

# Recent events table
st.subheader("📋 Recent Events")
events_df = recent_events()
if events_df.empty:
    st.info("No events yet. Start the tracker with `python main.py`.")
else:
    # Colour-code by event type
    def highlight_event(val):
        colours = {
            "ENTRY": "background-color: #1a3c1a; color: #a0f0a0",
            "EXIT": "background-color: #3c1a1a; color: #f0a0a0",
            "REGISTERED": "background-color: #1a2a3c; color: #a0c8f0",
            "RECOGNISED": "background-color: #2a2a1a; color: #f0f0a0",
        }
        return colours.get(val, "")

    st.dataframe(
        events_df.style.applymap(highlight_event, subset=["event_type"]),
        use_container_width=True,
    )

st.divider()

# Face images
img_col1, img_col2 = st.columns(2)

with img_col1:
    st.subheader("🟢 Latest Entries")
    entry_imgs = latest_images("entrie")  # folder is "entries"
    if entry_imgs:
        img_row = st.columns(min(3, len(entry_imgs)))
        for i, p in enumerate(entry_imgs[:3]):  # type: ignore
            img_row[i].image(p, use_column_width=True, caption=os.path.basename(p))
    else:
        st.caption("No entry images yet.")

with img_col2:
    st.subheader("🔴 Latest Exits")
    exit_imgs = latest_images("exit")
    if exit_imgs:
        img_row = st.columns(min(3, len(exit_imgs)))
        for i, p in enumerate(exit_imgs[:3]):  # type: ignore
            img_row[i].image(p, use_column_width=True, caption=os.path.basename(p))
    else:
        st.caption("No exit images yet.")

# Auto-refresh instructions
st.divider()
st.caption("ℹ️ Refresh the browser tab to see latest data. Run `streamlit run dashboard.py` separately from the tracker.")
