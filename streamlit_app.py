# -----------------------------
# LOOP — Dispatch + Pulse (all-in-one + quick wins, map + geo autofix)
# -----------------------------
import os
import io
import math
import json
import datetime as dt
import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path

# Optional .env (local dev convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------------
# OpenAI Client (new SDK)
# -----------------------------
from openai import OpenAI

def get_api_key():
    return (
        st.secrets.get("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

API_KEY = get_api_key()
if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY missing. Add it to .streamlit/secrets.toml or your environment/.env.")
    st.stop()

client = OpenAI(api_key=API_KEY)
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

def _combine_numeric(t: pd.DataFrame, candidates: list[str]):
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
    # consider swap if >50% invalid for lat and lng looks okay as lat
    if lat_bad / total > 0.5 and (lng_s.abs() <= 90).mean() > 0.5:
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

# ---------- Column mapping helpers (manual + auto) ----------
def map_columns_auto(rides_df, drivers_df):
    """Standardize likely column variants to common names (auto-detect)."""
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
    """Apply user-selected column mapping from st.session_state.colmap."""
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
    """Prefer manual mapping; fallback to auto-mapping. Fill missing std cols via autodetect."""
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
        # 🔧 Final geo sanity & auto-swap
    r_mapped = detect_geo_for_rides(r_mapped)
    d_mapped = detect_geo_for_drivers(d_mapped)

    # 🔧 Fallback IDs if none provided
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
    # geo if both sides have lat/lng
    if r is not None and d is not None:
        if {"pickup_lat","pickup_lng"}.issubset(r.columns) and {"lat","lng"}.issubset(d.columns):
            return "geo"
        # zone if any shared text keys appear on both
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
            # compute distances
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
            # match by any shared zone key we can find (priority: spot_id -> zone -> area)
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
    # nice ordering
    cols_front = ["ride_id","driver_id","driver_name","driver_phone","assignment_reason"]
    rest = [c for c in out.columns if c not in cols_front]
    return out[cols_front + rest]

# -----------------------------
# Pulse system + chat helpers
# -----------------------------
def system_prompt(memory_notes, uploads):
    mem_block = "\n".join([f"- {m['note']}" for m in memory_notes]) if memory_notes else "None."
    data_block = "\n".join([f"- {u['name']}: {u['snippet'][:500].strip()}" for u in uploads]) if uploads else "None."
    return f"""
You are {PULSE_NAME}, the AI copilot for LOOP (ride-hailing). You are the user's LITTLE BROTHER: warm, upbeat, razor-sharp, proactive.
Address the user as "{USER_NICK}" casually when appropriate.

Priorities: Safety > Reliability > Speed (sub-4-min pickups) > Cost-efficiency.
Use Operator Notes (memory) and Uploaded Reference Data if relevant.

Operator Notes:
{mem_block}

Uploaded Reference Data (high priority if relevant):
{data_block}

Rules:
- If greeted, reply like: "Hey Vish, I’m {PULSE_NAME} — I’m good! How are you?" then help with the task.
- If user says "remember: <note>" or "/remember <note>", acknowledge and store the note.
- When notes conflict, prefer the latest and confirm before discarding old policy.
- Keep responses concise and practical; ask clarifying questions when needed.
"""

def chat_complete(messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        top_p=1.0,
    )
    return resp.choices[0].message.content

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
                    text = f.read().decode("latin-1", errors="ignore")
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

        # Diagnostics (quiet helper)
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
                            if st.session_state.assignments_df is None or "ride_id" not in st.session_state.assignments_df.columns:
                                st.session_state.assignments_df = pd.DataFrame(columns=["ride_id","driver_id","driver_name","driver_phone","assignment_reason"])
                            st.session_state.assignments_df = st.session_state.assignments_df[
                                st.session_state.assignments_df["ride_id"].astype(str) != sel_ride
                            ]
                            st.session_state.assignments_df = pd.concat([st.session_state.assignments_df, pd.DataFrame([{
                                "ride_id": sel_ride,
                                "driver_id": drv.get("driver_id", drv.name),
                                "driver_name": drv.get("name",""),
                                "driver_phone": drv.get("phone",""),
                                "assignment_reason": f"nearest ({best[1]:.2f} km)"
                            }])], ignore_index=True)
                            st.success(f"Assigned ride {sel_ride} → driver {drv.get('driver_id', drv.name)}")
                        else:
                            st.warning("No nearby driver found within the max distance.")

        # ---------- Map View (sanitized to avoid vars()/to_numeric errors) ----------
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
            st.markdown(f"Hey {USER_NICK}, I’m {PULSE_NAME} — I’m good! How are you? 🙂")

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
                    reply = chat_complete(msgs)
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
