# ----------------------------- 
# LOOP — Dispatch + Pulse (all-in-one + quick wins, map + geo autofix, tool-calling)
# -----------------------------
import os
import io
import math
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
import pydeck as pdk
from dotenv import load_dotenv
from openai import OpenAI

# Load local env (for laptop runs)
load_dotenv()                 # reads .env if present
load_dotenv("openai.env")     # optional second env file

# Get key: prefer env; fall back to st.secrets ONLY if available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # works on Streamlit Cloud or local secrets.toml
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Put it in your local .env or in Streamlit Cloud Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = os.getenv("PULSE_MODEL", "gpt-4o-mini")

# -----------------------------
# Persistence paths (for quick-win: persistent memory)
# -----------------------------
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
MEMORY_PATH = DATA_DIR / "pulse_memory.json"

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def save_json(path, obj):
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Could not persist {path.name}: {e}")

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="LOOP — Dispatch + Pulse", page_icon="⚡", layout="wide")
st.title("⚡ LOOP — Dispatch + Pulse")

# -----------------------------
# Session State
# -----------------------------
if "rides_df" not in st.session_state: st.session_state.rides_df = None
if "drivers_df" not in st.session_state: st.session_state.drivers_df = None
if "assignments_df" not in st.session_state: st.session_state.assignments_df = None
if "messages" not in st.session_state: st.session_state.messages = []
if "pulse_memory" not in st.session_state: st.session_state.pulse_memory = load_json(MEMORY_PATH, [])  # persistent
if "uploaded_summaries" not in st.session_state: st.session_state.uploaded_summaries = []
# Column mapper state (quick win #1)
if "colmap" not in st.session_state:
    st.session_state.colmap = {
        "rides": {"ride_id": "ride_id", "lat": "", "lng": "", "zone": "", "area": "", "spot_id": ""},
        "drivers": {"driver_id": "driver_id", "name": "name", "phone": "phone", "lat": "", "lng": "", "zone": "", "area": "", "spot_id": ""}
    }

PULSE_NAME = "Pulse"
USER_NICK = "Vish"

# -----------------------------
# Utilities
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    # Return distance in km; robust to missing values
    try:
        for v in [lat1, lon1, lat2, lon2]:
            if v is None or v == "" or pd.isna(v):
                return None
        R = 6371.0
        p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
        dphi = math.radians(float(lat2)-float(lat1))
        dlmb = math.radians(float(lon2)-float(lon1))
        a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R*c
    except Exception:
        return None

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _first_match(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

# ---------- Geo autodetect & auto-fix ----------
LAT_ALIASES_RIDES = ["pickup_lat","pickup_latitude","lat","latitude","y"]
LNG_ALIASES_RIDES = ["pickup_lng","pickup_long","pickup_longitude","lng","lon","long","longitude","x"]
LAT_ALIASES_DRV   = ["lat","latitude","y","driver_lat","driver_latitude"]
LNG_ALIASES_DRV   = ["lng","lon","long","longitude","x","driver_lng","driver_lon"]

def _combine_numeric(t: pd.DataFrame, candidates: List[str]):
    """Return a single numeric Series built from first non-null across candidate columns."""
    series = None
    for col in candidates:
        if col in t.columns:
            col_data = t[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            col_num = pd.to_numeric(col_data, errors="coerce")
            series = col_num if series is None else series.fillna(col_num)
    return series

def _auto_swap_if_needed(lat_s: pd.Series, lng_s: pd.Series):
    """If most 'lat' values look like longitudes (|lat|>90) and most 'lng' look like lats, swap."""
    if lat_s is None or lng_s is None:
        return lat_s, lng_s
    total = max(int(lat_s.notna().sum()), 1)
    lat_bad = int((lat_s.abs() > 90).sum())
    if total > 0 and (lat_bad / total > 0.5) and (lng_s.abs() <= 90).mean() > 0.5:
        return lng_s, lat_s
    return lat_s, lng_s

def detect_geo_for_rides(r_df: pd.DataFrame):
    """Return rides DF with pickup_lat/pickup_lng added if possible (auto-detected)."""
    if r_df is None or r_df.empty: return r_df
    t = normalize_cols(r_df)
    lat = _combine_numeric(t, LAT_ALIASES_RIDES)
    lng = _combine_numeric(t, LNG_ALIASES_RIDES)
    lat, lng = _auto_swap_if_needed(lat, lng)
    if (lat is not None) and ("pickup_lat" not in t.columns):
        t["pickup_lat"] = lat
    if (lng is not None) and ("pickup_lng" not in t.columns):
        t["pickup_lng"] = lng
    return t

def detect_geo_for_drivers(d_df: pd.DataFrame):
    """Return drivers DF with lat/lng added if possible (auto-detected)."""
    if d_df is None or d_df.empty: return d_df
    t = normalize_cols(d_df)
    lat = _combine_numeric(t, LAT_ALIASES_DRV)
    lng = _combine_numeric(t, LNG_ALIASES_DRV)
    lat, lng = _auto_swap_if_needed(lat, lng)
    if (lat is not None) and ("lat" not in t.columns):
        t["lat"] = lat
    if (lng is not None) and ("lng" not in t.columns):
        t["lng"] = lng
    return t

# ---------- Column mapping helpers ----------
def map_columns_auto(rides_df, drivers_df):
    r = normalize_cols(rides_df)
    d = normalize_cols(drivers_df)
    info = {"rides": {}, "drivers": {}}

    # rides geo (autodetect)
    r = detect_geo_for_rides(r)
    # rides ids/zones
    rid = _first_match(r, ["ride_id","id","request_id"])
    if rid: r["ride_id"] = r[rid]; info["rides"]["ride_id"] = rid
    for std, cands in {
        "zone": ["zone","zone_id","zone_name","cluster","sector"],
        "area": ["area","area_id","area_name","neighborhood"],
        "spot_id": ["spot_id","spot","spot_name","stand_id","stand"],
    }.items():
        c = _first_match(r, cands)
        if c: r[std] = r[c]; info["rides"][std] = c

    # drivers geo (autodetect)
    d = detect_geo_for_drivers(d)
    # drivers ids/zones/meta
    did = _first_match(d, ["driver_id","id"])
    if did: d["driver_id"] = d[did]; info["drivers"]["driver_id"] = did
    name = _first_match(d, ["name","driver_name"])
    if name: d["name"] = d[name]; info["drivers"]["name"] = name
    phone = _first_match(d, ["phone","mobile","contact"])
    if phone: d["phone"] = d[phone]; info["drivers"]["phone"] = phone
    for std, cands in {
        "zone": ["zone","zone_id","zone_name","cluster","sector"],
        "area": ["area","area_id","area_name","neighborhood"],
        "spot_id": ["spot_id","spot","spot_name","stand_id","stand"],
    }.items():
        c = _first_match(d, cands)
        if c: d[std] = d[c]; info["drivers"][std] = c

    return r, d, info

def apply_manual_mapping(df, role):
    if df is None: return None
    df = normalize_cols(df)
    m = st.session_state.colmap.get(role, {})
    out = df.copy()

    if role == "rides":
        if m.get("ride_id") and m["ride_id"] in df.columns: out["ride_id"] = df[m["ride_id"]]
        if m.get("lat") and m["lat"] in df.columns: out["pickup_lat"] = pd.to_numeric(df[m["lat"]], errors="coerce")
        if m.get("lng") and m["lng"] in df.columns: out["pickup_lng"] = pd.to_numeric(df[m["lng"]], errors="coerce")
    if role == "drivers":
        if m.get("driver_id") and m["driver_id"] in df.columns: out["driver_id"] = df[m["driver_id"]]
        if m.get("name") and m["name"] in df.columns: out["name"] = df[m["name"]]
        if m.get("phone") and m["phone"] in df.columns: out["phone"] = df[m["phone"]]
        if m.get("lat") and m["lat"] in df.columns: out["lat"] = pd.to_numeric(df[m["lat"]], errors="coerce")
        if m.get("lng") and m["lng"] in df.columns: out["lng"] = pd.to_numeric(df[m["lng"]], errors="coerce")
    for key in ["zone","area","spot_id"]:
        src = m.get(key)
        if src and src in df.columns:
            out[key] = df[src]
    return out

def map_columns(rides_df, drivers_df):
    r_mapped = apply_manual_mapping(rides_df, "rides")
    d_mapped = apply_manual_mapping(drivers_df, "drivers")
    if r_mapped is None or d_mapped is None:
        r_auto, d_auto, _ = map_columns_auto(rides_df, drivers_df)
        return r_auto, d_auto, {"manual": False}

    # Fill any missing std cols via auto
    r_auto, d_auto, _ = map_columns_auto(rides_df, drivers_df)
    for k in ["pickup_lat","pickup_lng","zone","area","spot_id","ride_id"]:
        if k not in r_mapped.columns and k in r_auto.columns:
            r_mapped[k] = r_auto[k]
    for k in ["lat","lng","zone","area","spot_id","driver_id","name","phone"]:
        if k not in d_mapped.columns and k in d_auto.columns:
            d_mapped[k] = d_auto[k]

    # Final geo sanity & auto-swap
    r_mapped = detect_geo_for_rides(r_mapped)
    d_mapped = detect_geo_for_drivers(d_mapped)

    # Fallback IDs if none provided
    if r_mapped is not None and "ride_id" not in r_mapped.columns:
        r_mapped["ride_id"] = (
            r_mapped["request_id"].astype(str)
            if "request_id" in r_mapped.columns else
            r_mapped.index.astype(str)
        )
    if d_mapped is not None and "driver_id" not in d_mapped.columns:
        d_mapped["driver_id"] = d_mapped.index.astype(str)

    return r_mapped, d_mapped, {"manual": True}

def infer_assignment_mode(rides_df, drivers_df):
    r, d, _ = map_columns(rides_df, drivers_df)
    if r is not None and d is not None:
        if {"pickup_lat","pickup_lng"}.issubset(r.columns) and {"lat","lng"}.issubset(d.columns):
            return "geo"
        for key in ["zone","area","spot_id"]:
            if key in r.columns and key in d.columns:
                return "zone"
    return "none"

def auto_assign(rides_df, drivers_df, max_km=None):
    """
    Greedy one-to-one assignment.
    - If geo: pick nearest available driver (by distance).
    - Else if zone: match drivers with same zone/spot; otherwise leave unassigned.
    """
    r, d, info = map_columns(rides_df, drivers_df)
    if r is None or d is None:
        return pd.DataFrame()

    mode = infer_assignment_mode(r, d)

    # Clean geo columns if needed
    if mode == "geo":
        for c in ["pickup_lat","pickup_lng"]:
            r[c] = pd.to_numeric(r.get(c), errors="coerce")
        for c in ["lat","lng"]:
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        r = r.dropna(subset=["pickup_lat","pickup_lng"])
        d = d.dropna(subset=["lat","lng"])

    d = d.copy()
    d["assigned"] = False
    out_rows = []

    for idx, ride in r.iterrows():
        assignment = {"ride_id": ride.get("ride_id", idx)}
        # carry through ride fields
        for c in r.columns:
            assignment[f"ride_{c}"] = ride[c]

        chosen = None
        reason = None

        if mode == "geo":
            best = (None, 10**9)  # (idx, dist)
            for di, drv in d[~d["assigned"]].iterrows():
                dist = haversine(
                    ride.get("pickup_lat"), ride.get("pickup_lng"),
                    drv.get("lat"), drv.get("lng")
                )
                if dist is None:
                    continue
                if max_km is not None and dist > max_km:
                    continue
                if dist < best[1]:
                    best = (di, dist)
            if best[0] is not None:
                chosen = d.loc[best[0]]
                d.at[best[0], "assigned"] = True
                reason = f"nearest ({best[1]:.2f} km)"

        elif mode == "zone":
            for key in ["spot_id","zone","area"]:
                if key in r.columns and key in d.columns:
                    candidates = d[(~d["assigned"]) & (d[key].astype(str) == str(ride.get(key, "")))]
                    if len(candidates):
                        chosen = candidates.iloc[0]
                        d.at[chosen.name, "assigned"] = True
                        reason = f"zone match on {key}"
                        break
            if chosen is None:
                reason = "no same-zone driver available"

        else:
            reason = "no compatible columns; manual assignment needed"

        if chosen is not None:
            assignment.update({
                "driver_id": chosen.get("driver_id", chosen.name),
                "driver_name": chosen.get("name", ""),
                "driver_phone": chosen.get("phone",""),
                "driver_lat": chosen.get("lat",""),
                "driver_lng": chosen.get("lng",""),
                "assignment_reason": reason
            })
        else:
            assignment.update({
                "driver_id": "",
                "driver_name": "",
                "driver_phone": "",
                "driver_lat": "",
                "driver_lng": "",
                "assignment_reason": reason or "no match found"
            })

        out_rows.append(assignment)

    out = pd.DataFrame(out_rows)
    cols_front = ["ride_id","driver_id","driver_name","driver_phone","assignment_reason"]
    rest = [c for c in out.columns if c not in cols_front]
    return out[cols_front + rest]

# -----------------------------
# Quick-win: memory-driven "avoid" rules for spot recommendation
# -----------------------------
def extract_avoid_keywords(notes: List[Dict[str, Any]]) -> List[str]:
    """
    Very simple heuristic: from notes like "avoid SindhuBhavan at 6pm"
    we extract tokens after 'avoid'.
    """
    avoids = []
    for m in notes or []:
        t = m.get("note","")
        low = t.lower()
        if "avoid" in low:
            # crude parse: words after 'avoid' up to punctuation
            after = low.split("avoid", 1)[1].strip()
            token = after.split(",")[0].split(";")[0].split(" at ")[0].strip()
            if token:
                avoids.append(token)
    return avoids

def recommend_spot_for_driver(driver_id: str, radius_km: float = 3.0) -> Dict[str, Any]:
    """
    Heuristic: find the densest nearby ride pickup area for the given driver.
    - If geo available: compute nearest ride within radius and centroid of top-N close rides.
    - Respect memory 'avoid <keyword>' by skipping zones/areas/spots that mention those tokens.
    """
    r_mapped, d_mapped, _ = map_columns(st.session_state.rides_df, st.session_state.drivers_df)
    if r_mapped is None or d_mapped is None:
        return {"ok": False, "reason": "Upload rides and drivers first."}

    if "driver_id" not in d_mapped.columns:
        return {"ok": False, "reason": "driver_id not found in drivers."}

    rowd = d_mapped[d_mapped["driver_id"].astype(str) == str(driver_id)]
    if rowd.empty:
        return {"ok": False, "reason": f"Driver {driver_id} not found."}

    drv = rowd.iloc[0]
    if pd.isna(pd.to_numeric(drv.get("lat"), errors="coerce")) or pd.isna(pd.to_numeric(drv.get("lng"), errors="coerce")):
        # fall back to zone match
        for key in ["spot_id","zone","area"]:
            if key in d_mapped.columns and key in r_mapped.columns and pd.notna(drv.get(key,"")):
                cand = r_mapped[r_mapped[key].astype(str) == str(drv.get(key))]
                if not cand.empty:
                    return {
                        "ok": True,
                        "mode": "zone",
                        "zone_key": key,
                        "zone_value": str(drv.get(key)),
                        "reason": f"Same {key} as driver."
                    }
        return {"ok": False, "reason": "No geo for driver; no overlapping zone to recommend."}

    # Geo mode
    r_geo = detect_geo_for_rides(r_mapped)
    r_geo["pickup_lat"] = pd.to_numeric(r_geo.get("pickup_lat"), errors="coerce")
    r_geo["pickup_lng"] = pd.to_numeric(r_geo.get("pickup_lng"), errors="coerce")
    r_geo = r_geo.dropna(subset=["pickup_lat","pickup_lng"])
    if r_geo.empty:
        return {"ok": False, "reason": "No usable ride coordinates to analyze."}

    dlat = float(pd.to_numeric(drv["lat"], errors="coerce"))
    dlng = float(pd.to_numeric(drv["lng"], errors="coerce"))

    # Respect memory avoids
    avoids = extract_avoid_keywords(st.session_state.pulse_memory)
    def is_avoided(row) -> bool:
        hay = " ".join([str(row.get(k,"")) for k in ["spot_id","zone","area"] if k in r_geo.columns]).lower()
        return any(a in hay for a in avoids)

    r_geo["__dist"] = r_geo.apply(lambda row: haversine(dlat, dlng, row["pickup_lat"], row["pickup_lng"]), axis=1)
    nearby = r_geo[(r_geo["__dist"].notna()) & (r_geo["__dist"] <= float(radius_km))].copy()
    if not nearby.empty:
        # drop avoided
        if avoids:
            nearby = nearby[~nearby.apply(is_avoided, axis=1)]
        if not nearby.empty:
            # centroid of top 30 nearest
            nearby_sorted = nearby.sort_values("__dist").head(30)
            rec_lat = nearby_sorted["pickup_lat"].mean()
            rec_lng = nearby_sorted["pickup_lng"].mean()
            top_reason = f"{len(nearby_sorted)} recent pickups within {radius_km:.1f} km"
            if avoids:
                top_reason += f"; avoided: {', '.join(avoids)}"
            return {
                "ok": True,
                "mode": "geo",
                "lat": float(rec_lat),
                "lng": float(rec_lng),
                "samples": int(len(nearby_sorted)),
                "reason": top_reason
            }

    # If nothing nearby within radius, just pick closest ride centroid overall (still respect avoids)
    pool = r_geo.copy()
    if avoids:
        pool = pool[~pool.apply(is_avoided, axis=1)]
    if pool.empty:
        return {"ok": False, "reason": "All candidate spots are blocked by avoid rules."}
    pool = pool.sort_values("__dist").head(50)
    return {
        "ok": True,
        "mode": "geo-fallback",
        "lat": float(pool["pickup_lat"].mean()),
        "lng": float(pool["pickup_lng"].mean()),
        "samples": int(len(pool)),
        "reason": f"No nearby rides within {radius_km:.1f} km; suggested centroid of {len(pool)} closest pickups."
    }

# -----------------------------
# Pulse system + chat helpers (NOW WITH TOOL-CALLING)
# -----------------------------
def system_prompt(memory_notes, uploads):
    mem_block = "\n".join([f"- {m['note']}" for m in memory_notes]) if memory_notes else "None."
    data_block = "\n".join([f"- {u['name']}: {u['snippet'][:500].strip()}" for u in uploads]) if uploads else "None."
    return f"""
You are {PULSE_NAME}, the AI copilot for LOOP (ride-hailing). You are the user's LITTLE BROTHER: warm, upbeat, razor-sharp, proactive.
Call me "{USER_NICK}" casually when appropriate.

Priorities: Safety > Reliability > Speed (sub-4-min pickups) > Cost-efficiency.
Use Operator Notes (memory) and Uploaded Reference Data if relevant.

Operator Notes:
{mem_block}

Uploaded Reference Data (high priority if relevant):
{data_block}

Very important — when asked to do ANYTHING with rides, drivers, zones, assignments, or wait-spot recommendations,
use the provided TOOLS rather than guessing. Examples of when to call tools:
- “Auto-assign rides”, “Assign nearest for R123”, “What mode are we in?”, “Where should Ramesh wait now?”
- “Recommend a spot for driver D7 within 2 km”, “List unassigned rides”, “Why was R45 → D2?”

Rules:
- If greeted, reply like: "Hey {USER_NICK}, I’m {PULSE_NAME}! How are you doing today?" then help with the task.
- If user says "remember: <note>" or "/remember <note>", acknowledge and store the note.
- Keep responses concise and practical.
"""

def handle_inline_memory(text: str) -> bool:
    t = text.strip()
    low = t.lower()
    if low.startswith("remember:"):
        note = t[len("remember:"):].strip()
    elif low.startswith("/remember"):
        note = t[len("/remember"):].strip()
    else:
        return False
    if note:
        st.session_state.pulse_memory.append({
            "note": note,
            "added_at": dt.datetime.now().isoformat(timespec="seconds")
        })
        save_json(MEMORY_PATH, st.session_state.pulse_memory)  # persist
        with st.chat_message("assistant"):
            st.success(f"Got it. I’ll remember: _{note}_")
        return True
    return False

# ----- Tool schemas -----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_assignment_mode",
            "description": "Return the current assignment mode (geo, zone, or none) based on uploaded data.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_assign_all",
            "description": "Compute assignments for all rides using current mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_km": {"type": "number", "description": "Max distance (km) for geo mode; ignored in zone mode."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_nearest_for_ride",
            "description": "Assign nearest driver to a specific ride_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ride_id": {"type": "string"},
                    "max_km": {"type": "number"}
                },
                "required": ["ride_id"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "recommend_wait_spot",
        "description": "Recommend a waiting spot (lat,lng) for a driver using ID or name, based on nearby ride density and memory avoid rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "driver_id": {"type": "string"},
                "driver_name": {"type": "string"},
                "radius_km": {"type": "number", "default": 3.0}
            },
            "required": []
        }
    }
}

]

# ----- Tool dispatchers -----
def tool_get_assignment_mode() -> Dict[str, Any]:
    mode = infer_assignment_mode(st.session_state.rides_df, st.session_state.drivers_df)
    return {"mode": mode}

def tool_auto_assign_all(max_km: Optional[float]) -> Dict[str, Any]:
    mode = infer_assignment_mode(st.session_state.rides_df, st.session_state.drivers_df)
    out = auto_assign(st.session_state.rides_df, st.session_state.drivers_df, max_km=max_km if mode=="geo" else None)
    st.session_state.assignments_df = out
    return {"ok": True, "rows": int(len(out)), "mode": mode}

def tool_assign_nearest_for_ride(ride_id: str, max_km: Optional[float]) -> Dict[str, Any]:
    r_mapped, d_mapped, _ = map_columns(st.session_state.rides_df, st.session_state.drivers_df)
    if r_mapped is None or d_mapped is None or "ride_id" not in r_mapped.columns:
        return {"ok": False, "reason": "Upload rides and drivers first."}

    if not {"pickup_lat","pickup_lng"}.issubset(r_mapped.columns) or not {"lat","lng"}.issubset(d_mapped.columns):
        return {"ok": False, "reason": "Could not find usable coordinates for nearest assignment."}

    row = r_mapped[r_mapped["ride_id"].astype(str)==str(ride_id)]
    if row.empty:
        return {"ok": False, "reason": f"Ride {ride_id} not found."}
    row = row.iloc[0]
    rlat = pd.to_numeric(row["pickup_lat"], errors="coerce")
    rlng = pd.to_numeric(row["pickup_lng"], errors="coerce")
    ddf = d_mapped.copy()
    ddf["lat"] = pd.to_numeric(ddf["lat"], errors="coerce")
    ddf["lng"] = pd.to_numeric(ddf["lng"], errors="coerce")
    ddf = ddf.dropna(subset=["lat","lng"])
    best = (None, 10**9)
    for di, drv in ddf.iterrows():
        dist = haversine(rlat, rlng, drv["lat"], drv["lng"])
        if dist is None: continue
        if max_km and dist > max_km: continue
        if dist < best[1]: best = (di, dist)
    if best[0] is None:
        return {"ok": False, "reason": "No driver within limit."}
    drv = ddf.loc[best[0]]
    # upsert into assignments_df
    if st.session_state.assignments_df is None or "ride_id" not in getattr(st.session_state.assignments_df, "columns", []):
        st.session_state.assignments_df = pd.DataFrame(columns=["ride_id","driver_id","driver_name","driver_phone","assignment_reason"])
    st.session_state.assignments_df = st.session_state.assignments_df[
        st.session_state.assignments_df["ride_id"].astype(str) != str(ride_id)
    ]
    st.session_state.assignments_df = pd.concat([st.session_state.assignments_df, pd.DataFrame([{
        "ride_id": str(ride_id),
        "driver_id": drv.get("driver_id", drv.name),
        "driver_name": drv.get("name",""),
        "driver_phone": drv.get("phone",""),
        "assignment_reason": f"nearest ({best[1]:.2f} km)"
    }])], ignore_index=True)
    return {
        "ok": True,
        "ride_id": str(ride_id),
        "driver_id": str(drv.get("driver_id", drv.name)),
        "distance_km": float(best[1])
    }

def tool_recommend_wait_spot(driver_id: Optional[str], driver_name: Optional[str], radius_km: Optional[float]) -> Dict[str, Any]:
    """
    Name/ID resolver for recommending a wait spot.
    Investor-friendly behavior:
      - Accepts driver_id OR driver_name
      - Matching order: Exact (case-insensitive) → Prefix → Space-insensitive contains → Fuzzy (SequenceMatcher)
      - If multiple likely matches, return a short disambiguation list
    """

    # ---- helpers
    def norm(s: str) -> str:
        return str(s or "").strip().lower()

    def nospace(s: str) -> str:
        return norm(s).replace(" ", "")

    def token_sort(s: str) -> str:
        toks = [t for t in norm(s).split() if t]
        return " ".join(sorted(toks))

    def fuzzy_ratio(a: str, b: str) -> float:
        # robust typo tolerance using stdlib (no extra deps)
        return SequenceMatcher(None, norm(a), norm(b)).ratio()

    if (not driver_id) and driver_name:
        try:
            # Standardize driver data
            _, d_mapped, _ = map_columns(st.session_state.rides_df, st.session_state.drivers_df)
            if d_mapped is None or d_mapped.empty or "name" not in d_mapped.columns or "driver_id" not in d_mapped.columns:
                return {"ok": False, "reason": "Drivers data not loaded or missing name/driver_id columns."}

            needle = str(driver_name)
            needle_n = norm(needle)
            needle_ns = nospace(needle)
            needle_ts = token_sort(needle)

            # Build a small scoring table
            rows = []
            for i, row in d_mapped.iterrows():
                name = str(row.get("name", ""))
                did  = str(row.get("driver_id", ""))
                name_n  = norm(name)
                name_ns = nospace(name)
                name_ts = token_sort(name)

                score = 0.0
                reason = ""

                # 1) Exact (case-insensitive or token-sorted exact)
                if name_n == needle_n or name_ts == needle_ts:
                    score = 1.0
                    reason = "exact"
                # 2) Prefix (common UX: user types first name or start of it)
                elif name_n.startswith(needle_n) or name_ts.startswith(needle_ts):
                    score = 0.92
                    reason = "prefix"
                # 3) Space-insensitive contains ("rkesh" vs "rakesh", "mera" vs "meera shah")
                elif needle_ns and (needle_ns in name_ns):
                    score = 0.86
                    reason = "contains"
                else:
                    # 4) Fuzzy (typo-tolerant) — SequenceMatcher
                    fr = fuzzy_ratio(needle, name)  # 0..1
                    score = fr
                    reason = f"fuzzy:{fr:.2f}"

                rows.append({"driver_id": did, "name": name, "score": float(score), "why": reason})

            if not rows:
                return {"ok": False, "reason": f"No drivers loaded to match against '{driver_name}'."}

            # Sort by score descending
            rows.sort(key=lambda r: r["score"], reverse=True)

            # Thresholds: accept single strong match; otherwise ask
            BEST_ACCEPT = 0.75   # accept automatically above this
            AMBIGUOUS_BAND = 0.72  # near-threshold: show options if multiple are close

            top = rows[0]
            # If top is clearly best and unique enough, take it
            if top["score"] >= BEST_ACCEPT:
                # If second is very close (within 0.03), prefer disambiguation to be safe in demo
                second = rows[1] if len(rows) > 1 else None
                if second and (top["score"] - second["score"] < 0.03):
                    choices = [{"driver_id": r["driver_id"], "name": r["name"], "score": round(r["score"], 2)} for r in rows[:6]]
                    return {"ok": False, "reason": f"Multiple close matches for '{driver_name}'. Please specify ID.", "choices": choices}
                driver_id = top["driver_id"]
            else:
                # Check if a few are in the ambiguous band; if exactly one, take it; else ask
                near = [r for r in rows if r["score"] >= AMBIGUOUS_BAND]
                if len(near) == 1:
                    driver_id = near[0]["driver_id"]
                elif len(near) > 1:
                    choices = [{"driver_id": r["driver_id"], "name": r["name"], "score": round(r["score"], 2)} for r in near[:6]]
                    return {"ok": False, "reason": f"Multiple drivers match '{driver_name}'. Please specify ID.", "choices": choices}
                else:
                    return {"ok": False, "reason": f"No driver found close to '{driver_name}'. Try a fuller name or the ID."}

        except Exception as e:
            return {"ok": False, "reason": f"Name resolution error: {e}"}

    if not driver_id:
        return {"ok": False, "reason": "Please provide a driver_id or a driver_name."}

    # Final: call the core spot recommendation
    return recommend_spot_for_driver(driver_id, radius_km or 3.0)

    """
    Enhanced version:
    - Accepts driver_id OR driver_name
    - Prioritizes exact matches before partial matches
    - Fuzzy fallback (typo tolerance) without extra libraries
    - Returns top match or asks for clarification if duplicates
    """

    if (not driver_id) and driver_name:
        try:
            # Standardize driver data
            _, d_mapped, _ = map_columns(st.session_state.rides_df, st.session_state.drivers_df)
            if d_mapped is None or d_mapped.empty or "name" not in d_mapped.columns or "driver_id" not in d_mapped.columns:
                return {"ok": False, "reason": "Drivers data not loaded or missing name/driver_id columns."}

            name_series = d_mapped["name"].astype(str)
            needle = str(driver_name).strip()

            # 1️⃣ Exact match (case-insensitive)
            exact_matches = d_mapped[name_series.str.lower() == needle.lower()]
            if len(exact_matches) == 1:
                driver_id = str(exact_matches.iloc[0]["driver_id"])
            elif len(exact_matches) > 1:
                choices = exact_matches[["driver_id", "name"]].astype(str).to_dict("records")
                return {"ok": False, "reason": f"Multiple exact matches for '{driver_name}'. Please specify ID.", "choices": choices}

            # 2️⃣ Partial match if no exact match
            if not driver_id:
                partial_matches = d_mapped[name_series.str.lower().str.replace(" ", "").str.contains(needle.lower().replace(" ", ""), na=False)]
                if len(partial_matches) == 1:
                    driver_id = str(partial_matches.iloc[0]["driver_id"])
                elif len(partial_matches) > 1:
                    choices = partial_matches[["driver_id", "name"]].astype(str).head(10).to_dict("records")
                    return {"ok": False, "reason": f"Multiple drivers match '{driver_name}'. Please specify an ID.", "choices": choices}

            # 3️⃣ Fuzzy match if still no match (very lightweight)
            if not driver_id:
                def simple_ratio(a, b):
                    a, b = a.lower(), b.lower()
                    matches = sum(1 for x, y in zip(a, b) if x == y)
                    return matches / max(len(a), len(b))
                fuzzy_scores = name_series.apply(lambda x: simple_ratio(needle, str(x)))
                best_idx = fuzzy_scores.idxmax()
                if fuzzy_scores[best_idx] >= 0.6:  # 60% similarity threshold
                    driver_id = str(d_mapped.loc[best_idx, "driver_id"])

            if not driver_id:
                return {"ok": False, "reason": f"No driver found matching '{driver_name}'."}

        except Exception as e:
            return {"ok": False, "reason": f"Name resolution error: {e}"}

    if not driver_id:
        return {"ok": False, "reason": "Please provide a driver_id or a driver_name."}

    # Final: call the core spot recommendation
    return recommend_spot_for_driver(driver_id, radius_km or 3.0)

# Map tool name → handler
TOOL_IMPL = {
    "get_assignment_mode": lambda **_: tool_get_assignment_mode(),
    "auto_assign_all": lambda max_km=None, **_: tool_auto_assign_all(max_km),
    "assign_nearest_for_ride": lambda ride_id, max_km=None, **_: tool_assign_nearest_for_ride(ride_id, max_km),
    "recommend_wait_spot": lambda driver_id=None, driver_name=None, radius_km=3.0, **_: tool_recommend_wait_spot(driver_id, driver_name, radius_km),

}

def run_pulse_with_tools(history_messages: List[Dict[str, str]]) -> str:
    """
    1) Send to model with tool schemas
    2) If tool calls returned, execute each and append tool results
    3) Send back to model for final natural-language answer
    """
    try:
        first = client.chat.completions.create(
            model=MODEL,
            messages=history_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.4,
            top_p=1.0,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI error (first call): {e}")

    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        # Append the assistant's tool call message
        history_messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in tool_calls]
        })

        # Execute tool calls
        for tc in tool_calls:
            name = tc.function.name
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            impl = TOOL_IMPL.get(name)
            if impl is None:
                tool_result = {"ok": False, "error": f"Tool {name} not implemented."}
            else:
                try:
                    tool_result = impl(**args)
                except Exception as e:
                    tool_result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

            history_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        # Finalization call for a natural-language answer
        try:
            final = client.chat.completions.create(
                model=MODEL,
                messages=history_messages,
                temperature=0.3,
                top_p=1.0,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI error (finalization): {e}")
        return final.choices[0].message.content or "(no reply)"
    else:
        # No tool use needed
        return msg.content or "(no reply)"

# -----------------------------
# Sidebar — Memory + References (persistent memory)
# -----------------------------
with st.sidebar:
    st.subheader("🧠 Pulse Memory")
    new_note = st.text_area("Add a note", placeholder="e.g., 6pm—avoid SindhuBhavan; politician visit = heavy traffic")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Save"):
            if new_note.strip():
                st.session_state.pulse_memory.append({
                    "note": new_note.strip(),
                    "added_at": dt.datetime.now().isoformat(timespec="seconds")
                })
                save_json(MEMORY_PATH, st.session_state.pulse_memory)  # persist
                st.success("Saved.")
    with c2:
        if st.button("🗑️ Clear"):
            st.session_state.pulse_memory = []
            save_json(MEMORY_PATH, st.session_state.pulse_memory)  # persist
            st.success("Cleared.")

    if st.session_state.pulse_memory:
        st.markdown("**Current notes:**")
        for i, m in enumerate(st.session_state.pulse_memory, 1):
            st.markdown(f"- {i}. {m['note']} _(added {m['added_at']})_")

    st.markdown("---")
    st.subheader("📎 Reference Files")
    ref_files = st.file_uploader("Upload txt/csv/pdf", type=["txt","csv","pdf"], accept_multiple_files=True)
    if ref_files:
        for f in ref_files:
            try:
                if f.type in ("text/plain","text/csv"):
                    text = f.read().decode("utf-8", errors="ignore")
                elif f.type == "application/pdf":
                    text = f.read().decode("latin-1", errors="ignore")  # lightweight fallback
                else:
                    text = ""
                snippet = text[:30000]
                st.session_state.uploaded_summaries.append({"name": f.name, "snippet": snippet})
                st.success(f"Ingested: {f.name}")
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

# -----------------------------
# Tabs: Dispatch | Pulse Chat | Data
# -----------------------------
tab_dispatch, tab_chat, tab_data = st.tabs(["🧭 Dispatch", "💬 Pulse Chat", "📊 Data"])

# ======== DISPATCH ========
with tab_dispatch:
    st.subheader("Upload & Assign")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        rides_file = st.file_uploader("Upload rides.csv", type=["csv"], key="rides_uploader")
        if rides_file is not None:
            try:
                st.session_state.rides_df = pd.read_csv(rides_file)
                st.success("Rides loaded.")
            except Exception as e:
                st.error(f"Failed to read rides: {e}")

        st.caption("Expected ride columns (any subset works): ride_id, pickup_lat, pickup_lng, zone/area/spot_id, pickup_time, notes")

    with col_u2:
        drivers_file = st.file_uploader("Upload drivers.csv", type=["csv"], key="drivers_uploader")
        if drivers_file is not None:
            try:
                st.session_state.drivers_df = pd.read_csv(drivers_file)
                st.success("Drivers loaded.")
            except Exception as e:
                st.error(f"Failed to read drivers: {e}")

        st.caption("Expected driver columns: driver_id, name, phone, lat, lng, zone/area/spot_id, available")

    # ---------- Manual Column Mapper UI ----------
    st.markdown("### 🧩 Column Mapper (Optional but recommended)")
    if st.session_state.rides_df is not None or st.session_state.drivers_df is not None:
        with st.expander("Map Ride Columns"):
            if st.session_state.rides_df is not None:
                rcols = [""] + list(normalize_cols(st.session_state.rides_df).columns)
                rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
                with rc1:
                    st.session_state.colmap["rides"]["ride_id"] = st.selectbox("ride_id", rcols, index=rcols.index(st.session_state.colmap["rides"].get("ride_id","")) if st.session_state.colmap["rides"].get("ride_id","") in rcols else 0)
                with rc2:
                    st.session_state.colmap["rides"]["lat"] = st.selectbox("pickup_lat", rcols, index=rcols.index(st.session_state.colmap["rides"].get("lat","")) if st.session_state.colmap["rides"].get("lat","") in rcols else 0)
                with rc3:
                    st.session_state.colmap["rides"]["lng"] = st.selectbox("pickup_lng", rcols, index=rcols.index(st.session_state.colmap["rides"].get("lng","")) if st.session_state.colmap["rides"].get("lng","") in rcols else 0)
                with rc4:
                    st.session_state.colmap["rides"]["zone"] = st.selectbox("zone", rcols, index=rcols.index(st.session_state.colmap["rides"].get("zone","")) if st.session_state.colmap["rides"].get("zone","") in rcols else 0)
                with rc5:
                    st.session_state.colmap["rides"]["area"] = st.selectbox("area", rcols, index=rcols.index(st.session_state.colmap["rides"].get("area","")) if st.session_state.colmap["rides"].get("area","") in rcols else 0)
                with rc6:
                    st.session_state.colmap["rides"]["spot_id"] = st.selectbox("spot_id", rcols, index=rcols.index(st.session_state.colmap["rides"].get("spot_id","")) if st.session_state.colmap["rides"].get("spot_id","") in rcols else 0)
            else:
                st.info("Upload rides.csv to map ride columns.")

        with st.expander("Map Driver Columns"):
            if st.session_state.drivers_df is not None:
                dcols = [""] + list(normalize_cols(st.session_state.drivers_df).columns)
                dc1, dc2, dc3, dc4, dc5, dc6, dc7 = st.columns(7)
                with dc1:
                    st.session_state.colmap["drivers"]["driver_id"] = st.selectbox("driver_id", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("driver_id","")) if st.session_state.colmap["drivers"].get("driver_id","") in dcols else 0)
                with dc2:
                    st.session_state.colmap["drivers"]["name"] = st.selectbox("name", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("name","")) if st.session_state.colmap["drivers"].get("name","") in dcols else 0)
                with dc3:
                    st.session_state.colmap["drivers"]["phone"] = st.selectbox("phone", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("phone","")) if st.session_state.colmap["drivers"].get("phone","") in dcols else 0)
                with dc4:
                    st.session_state.colmap["drivers"]["lat"] = st.selectbox("lat", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("lat","")) if st.session_state.colmap["drivers"].get("lat","") in dcols else 0)
                with dc5:
                    st.session_state.colmap["drivers"]["lng"] = st.selectbox("lng", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("lng","")) if st.session_state.colmap["drivers"].get("lng","") in dcols else 0)
                with dc6:
                    st.session_state.colmap["drivers"]["zone"] = st.selectbox("zone", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("zone","")) if st.session_state.colmap["drivers"].get("zone","") in dcols else 0)
                with dc7:
                    st.session_state.colmap["drivers"]["spot_id"] = st.selectbox("spot_id", dcols, index=dcols.index(st.session_state.colmap["drivers"].get("spot_id","")) if st.session_state.colmap["drivers"].get("spot_id","") in dcols else 0)
                st.caption("Tip: set either (lat,lng) for geo or (zone/spot_id/area) for zone matching.")
            else:
                st.info("Upload drivers.csv to map driver columns.")

    st.markdown("---")

    if st.session_state.rides_df is not None and st.session_state.drivers_df is not None:
        cA, cB, cC = st.columns([1,1,1])
        with cA:
            mode = infer_assignment_mode(st.session_state.rides_df, st.session_state.drivers_df)
            st.info(f"Detected assignment mode: **{mode}**")
        with cB:
            max_km = st.number_input("Max distance (km) for assignment (geo mode only)", min_value=0.0, value=8.0, step=0.5)
        with cC:
            if st.button("🚀 Auto-Assign"):
                st.session_state.assignments_df = auto_assign(
                    st.session_state.rides_df,
                    st.session_state.drivers_df,
                    max_km=max_km if mode=="geo" else None
                )
                st.success("Assignments computed.")

        # ---------- Assign Nearest (single ride) ----------
        st.markdown("#### 🎯 Assign Nearest (single ride)")
        r_mapped, d_mapped, _ = map_columns(st.session_state.rides_df, st.session_state.drivers_df)

        with st.expander("🔎 Assign-Nearest Diagnostics", expanded=False):
            have_r = r_mapped is not None and {"pickup_lat","pickup_lng"}.issubset(set(r_mapped.columns))
            have_d = d_mapped is not None and {"lat","lng"}.issubset(set(d_mapped.columns))
            st.write("Rides have pickup_lat/pickup_lng:", have_r)
            st.write("Drivers have lat/lng:", have_d)
            if have_r: st.write(r_mapped[["pickup_lat","pickup_lng"]].head(3))
            if have_d: st.write(d_mapped[["lat","lng"]].head(3))

        ride_choices = r_mapped["ride_id"].astype(str).tolist() if (r_mapped is not None and "ride_id" in r_mapped.columns) else []
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            sel_ride = st.selectbox("Pick a ride", ride_choices) if ride_choices else None
        with c2:
            max_km_single = st.number_input("Max km (single)", min_value=0.0, value=8.0, step=0.5, key="single_km")
        with c3:
            if st.button("⚡ Assign Nearest"):
                if r_mapped is None or d_mapped is None or "ride_id" not in r_mapped.columns:
                    st.warning("Upload rides and drivers first.")
                elif not {"pickup_lat","pickup_lng"}.issubset(r_mapped.columns) or not {"lat","lng"}.issubset(d_mapped.columns):
                    st.warning("Could not find usable coordinates after auto-detect. Check Diagnostics for columns/values.")
                elif not sel_ride:
                    st.warning("Pick a ride first.")
                else:
                    row = r_mapped[r_mapped["ride_id"].astype(str)==sel_ride]
                    if row.empty:
                        st.warning("Selected ride not found.")
                    else:
                        row = row.iloc[0]
                        rlat = pd.to_numeric(row["pickup_lat"], errors="coerce")
                        rlng = pd.to_numeric(row["pickup_lng"], errors="coerce")
                        ddf = d_mapped.copy()
                        ddf["lat"] = pd.to_numeric(ddf["lat"], errors="coerce")
                        ddf["lng"] = pd.to_numeric(ddf["lng"], errors="coerce")
                        ddf = ddf.dropna(subset=["lat","lng"])
                        best = (None, 10**9)
                        for di, drv in ddf.iterrows():
                            dist = haversine(rlat, rlng, drv["lat"], drv["lng"])
                            if dist is None: continue
                            if max_km_single and dist > max_km_single: continue
                            if dist < best[1]: best = (di, dist)
                        if best[0] is not None:
                            drv = ddf.loc[best[0]]
                            if st.session_state.assignments_df is None or "ride_id" not in getattr(st.session_state.assignments_df, "columns", []):
                                st.session_state.assignments_df = pd.DataFrame(columns=["ride_id","driver_id","driver_name","driver_phone","assignment_reason"])
                            st.session_state.assignments_df = st.session_state.assignments_df[
                                st.session_state.assignments_df["ride_id"].astype(str) != str(sel_ride)
                            ]
                            st.session_state.assignments_df = pd.concat([st.session_state.assignments_df, pd.DataFrame([{
                                "ride_id": str(sel_ride),
                                "driver_id": drv.get("driver_id", drv.name),
                                "driver_name": drv.get("name",""),
                                "driver_phone": drv.get("phone",""),
                                "assignment_reason": f"nearest ({best[1]:.2f} km)"
                            }])], ignore_index=True)
                            st.success(f"Assigned ride {sel_ride} → driver {drv.get('driver_id', drv.name)}")
                        else:
                            st.warning("No nearby driver found within the max distance.")

        # ---------- Map View ----------
        st.markdown("### 🗺️ Map View")

        def safe_map_df(df, role="rides"):
            """Build clean lat/lng for map from many possible columns; drop NaNs; return minimal DF."""
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame(columns=["lat","lng"])
            t = df.copy()
            t = normalize_cols(t)
            t = t.loc[:, ~t.columns.duplicated()]
            if role == "rides":
                t = detect_geo_for_rides(t)
                lat_series = pd.to_numeric(t.get("pickup_lat"), errors="coerce")
                lng_series = pd.to_numeric(t.get("pickup_lng"), errors="coerce")
            else:
                t = detect_geo_for_drivers(t)
                lat_series = pd.to_numeric(t.get("lat"), errors="coerce")
                lng_series = pd.to_numeric(t.get("lng"), errors="coerce")
            out = pd.DataFrame({"lat": lat_series, "lng": lng_series}).dropna(subset=["lat","lng"])
            return out

        r_mapped, d_mapped, _ = map_columns(st.session_state.rides_df, st.session_state.drivers_df)
        rides_map_df = safe_map_df(r_mapped, "rides")
        drivers_map_df = safe_map_df(d_mapped, "drivers")

        layers = []
        if not rides_map_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=rides_map_df.to_dict("records"),
                get_position='[lng, lat]',
                get_radius=50,
                pickable=True,
            ))
        if not drivers_map_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=drivers_map_df.to_dict("records"),
                get_position='[lng, lat]',
                get_radius=60,
                pickable=True,
            ))

        # Lines for assignments if geo data available
        if st.session_state.assignments_df is not None and len(st.session_state.assignments_df):
            assign = st.session_state.assignments_df.copy()
            if {"ride_id","driver_id"}.issubset(assign.columns):
                # Prepare ride coords
                if r_mapped is not None and {"ride_id"}.issubset(r_mapped.columns):
                    r_tmp = detect_geo_for_rides(r_mapped)
                    join_rides = pd.DataFrame({
                        "ride_id": r_tmp["ride_id"].astype(str),
                        "pickup_lat": pd.to_numeric(r_tmp.get("pickup_lat"), errors="coerce"),
                        "pickup_lng": pd.to_numeric(r_tmp.get("pickup_lng"), errors="coerce"),
                    }).dropna(subset=["pickup_lat","pickup_lng"])
                    assign["ride_id"] = assign["ride_id"].astype(str)
                    assign = assign.merge(join_rides, on="ride_id", how="left")
                # Prepare driver coords
                if d_mapped is not None and {"driver_id"}.issubset(d_mapped.columns):
                    d_tmp = detect_geo_for_drivers(d_mapped)
                    join_drivers = pd.DataFrame({
                        "driver_id": d_tmp["driver_id"],
                        "lat": pd.to_numeric(d_tmp.get("lat"), errors="coerce"),
                        "lng": pd.to_numeric(d_tmp.get("lng"), errors="coerce"),
                    }).dropna(subset=["lat","lng"])
                    assign = assign.merge(join_drivers, on="driver_id", how="left")

                # Build lines
                if {"pickup_lat","pickup_lng","lat","lng"}.issubset(assign.columns):
                    for c in ["pickup_lat","pickup_lng","lat","lng"]:
                        assign[c] = pd.to_numeric(assign[c], errors="coerce")
                    assign_lines = assign.dropna(subset=["pickup_lat","pickup_lng","lat","lng"])
                    if len(assign_lines):
                        line_records = [
                            {
                                "from_lng": float(row["pickup_lng"]),
                                "from_lat": float(row["pickup_lat"]),
                                "to_lng": float(row["lng"]),
                                "to_lat": float(row["lat"]),
                            }
                            for _, row in assign_lines.iterrows()
                        ]
                        layers.append(pdk.Layer(
                            "LineLayer",
                            data=line_records,
                            get_source_position='[from_lng, from_lat]',
                            get_target_position='[to_lng, to_lat]',
                            pickable=True,
                            width_min_pixels=2,
                        ))

        # Initial view
        if not rides_map_df.empty:
            center_lat = rides_map_df["lat"].mean()
            center_lng = rides_map_df["lng"].mean()
        elif not drivers_map_df.empty:
            center_lat = drivers_map_df["lat"].mean()
            center_lng = drivers_map_df["lng"].mean()
        else:
            # Ahmedabad default
            center_lat, center_lng = 23.0225, 72.5714

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=11)
        st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers))

        # ---------- Review & export ----------
        st.markdown("### Review & Edit Assignments")
        if st.session_state.assignments_df is not None:
            editable = st.session_state.assignments_df.copy()
            editable = st.data_editor(
                editable,
                num_rows="dynamic",
                use_container_width=True,
                key="assign_editor",
                hide_index=True
            )
            st.session_state.assignments_df = editable

            # Download
            csv_buf = io.StringIO()
            editable.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download assignments.csv",
                data=csv_buf.getvalue(),
                file_name="assignments.csv",
                mime="text/csv"
            )
        else:
            st.info("Run Auto-Assign or use 'Assign Nearest' to create assignments.")

    else:
        st.warning("Upload both rides.csv and drivers.csv to enable assignment and map.")

# ======== PULSE CHAT ========
with tab_chat:
    st.subheader("Pulse — AI Ops Copilot")
    # Seed a friendly first message (visual)
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(f"Hey {USER_NICK}, I’m {PULSE_NAME}! How are you doing today? 🙂")

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Type a message… (tip: 'remember: 6pm avoid SindhuBhavan')")
    if user_input:
        # inline memory command?
        if handle_inline_memory(user_input):
            pass
        else:
            st.session_state.messages.append({"role":"user","content": user_input})
            msgs = [
                {"role":"system", "content": system_prompt(st.session_state.pulse_memory, st.session_state.uploaded_summaries)}
            ] + st.session_state.messages[-20:]

            with st.chat_message("assistant"):
                try:
                    reply = run_pulse_with_tools(msgs)
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    reply = "Hit a snag calling the model. Check your API key/usage and try again."
                st.markdown(reply)
                st.session_state.messages.append({"role":"assistant","content": reply})

# ======== DATA ========
with tab_data:
    st.subheader("Data Preview")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Rides**")
        if st.session_state.rides_df is not None:
            st.dataframe(st.session_state.rides_df, use_container_width=True)
        else:
            st.info("Upload rides.csv in the Dispatch tab.")

    with col_d2:
        st.markdown("**Drivers**")
        if st.session_state.drivers_df is not None:
            st.dataframe(st.session_state.drivers_df, use_container_width=True)
        else:
            st.info("Upload drivers.csv in the Dispatch tab.")

    st.markdown("---")
    st.markdown("**Assignments**")
    if st.session_state.assignments_df is not None:
        st.dataframe(st.session_state.assignments_df, use_container_width=True)
    else:
        st.info("Run Auto-Assign in the Dispatch tab.")
