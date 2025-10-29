#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------- CONFIG ----------
DEFAULT_XLSX = Path("data/ofe_list_with_coords.xlsx")
DEFAULT_SHEET = 0
COUNTRY_BIAS = "Australia"
USER_AGENT = "ofe-nearest/1.5 (research; contact: youremail@example.com)"
FEATURE_KEYS = [str(i) for i in range(1, 16)]  # "1".."15"
# ----------------------------

LABELS = {
    "council": "Council",
    "name": "Name",
    "park": "Park Name",
    "address": "Address",
    "link": "Link",
    "confirmed": "Visually confirmed OFE?",
    "searchok": "Does search function on website return correct results?",
}

# ================= Utilities =================
def load_with_auto_header(src: Path, sheet) -> pd.DataFrame:
    """Load Excel and detect the header row by finding the first row containing 'Address'."""
    raw = pd.read_excel(src, sheet_name=sheet, header=None, dtype=str)
    header_row = None
    for i in range(min(10, len(raw))):
        row_vals = raw.iloc[i].astype(str).fillna("")
        if any(re.search(r"\baddress\b", str(v), re.I) for v in row_vals):
            header_row = i
            break
    if header_row is None:
        header_row = 0
    cols = raw.iloc[header_row].fillna("").astype(str).tolist()
    clean_cols, seen = [], {}
    for c in cols:
        c2 = re.sub(r"\s+", " ", c.strip()) or "Unnamed"
        seen[c2] = seen.get(c2, 0) + 1
        if seen[c2] > 1:
            c2 = f"{c2}_{seen[c2]}"
        clean_cols.append(c2)
    df = raw.iloc[header_row + 1:].copy()
    df.columns = clean_cols
    df = df.reset_index(drop=True)
    return df

def normalize(name: str) -> str:
    n = re.sub(r"[\s_()\-]+", "", name.strip().lower())
    n = n.replace("longitud", "lon").replace("longitude", "lon").replace("long", "lon").replace("lng", "lon")
    n = n.replace("latitude", "lat")
    return n

def find_latlon_columns(df: pd.DataFrame):
    cmap = {normalize(c): c for c in df.columns}
    lat = next((cmap[k] for k in cmap if k.startswith("lat")), None)
    lon = next((cmap[k] for k in cmap if k.startswith("lon")), None)
    return lat, lon

def col_like(df, name):
    for c in df.columns:
        if c.lower().strip() == name.lower():
            return c
    return None

def build_name(row, cols_map):
    # Prefer Park Name, then Name
    for key in ["park", "name"]:
        col = cols_map.get(key)
        if col and pd.notna(row.get(col)) and str(row.get(col)).strip():
            return str(row.get(col)).strip()
    return "Outdoor Fitness Site"

def icon_color(val) -> str:
    if val is None or str(val).strip() == "":
        return "blue"
    v = str(val).strip().lower()
    if v in {"yes", "y", "true", "1"}:
        return "green"
    if v in {"no", "n", "false", "0"}:
        return "red"
    return "blue"

def popup_html(row, cols_map) -> str:
    parts = [f"<b>{build_name(row, cols_map)}</b>"]
    for key, label in LABELS.items():
        col = cols_map.get(key)
        if key in {"name", "park"}:
            continue
        val = row.get(col) if col else None
        if pd.notna(val) and str(val).strip():
            parts.append(f"{label}: {val}")
    feats = [str(row[c]).strip() for c in cols_map.get("features", []) if pd.notna(row.get(c)) and str(row.get(c)).strip()]
    if feats:
        parts.append("Features: " + ", ".join(feats))
    link_col = cols_map.get("link")
    link = row.get(link_col) if link_col else None
    if pd.notna(link) and str(link).strip():
        parts.append(f'<a href="{str(link).strip()}" target="_blank">Open council/park page</a>')
    return "<br>".join(parts)

@st.cache_data(show_spinner=False)
def geocode_cached(addr: str) -> Optional[Tuple[float, float, str]]:
    geo = Nominatim(user_agent=USER_AGENT, timeout=20)
    geocode = RateLimiter(geo.geocode, min_delay_seconds=1.1, swallow_exceptions=True)
    q = addr if COUNTRY_BIAS.lower() in addr.lower() else f"{addr}, {COUNTRY_BIAS}"
    loc = geocode(q)
    return (loc.latitude, loc.longitude, loc.address) if loc else None

# ================= Main App =================
st.set_page_config(page_title="OFE Nearest Finder", layout="wide")
st.title("Outdoor Fitness Equipment Finder")

# Persist across reruns
if "user_loc" not in st.session_state:
    st.session_state.user_loc = None
if "geocoded_text" not in st.session_state:
    st.session_state.geocoded_text = None

# Load bundled data
if not DEFAULT_XLSX.exists():
    st.error(f"File not found: {DEFAULT_XLSX.resolve()}")
    st.stop()

df = load_with_auto_header(DEFAULT_XLSX, DEFAULT_SHEET)
cols_map = {
    "council":   col_like(df, "Council"),
    "name":      col_like(df, "Name"),
    "park":      col_like(df, "Park Name"),
    "address":   col_like(df, "Address"),
    "link":      col_like(df, "Link"),
    "confirmed": col_like(df, "Visually confirmed OFE?"),
    "searchok":  col_like(df, "Does search function on website return correct results?"),
    "features":  [c for c in df.columns if c.strip() in FEATURE_KEYS]
}

lat_col, lon_col = find_latlon_columns(df)
if not lat_col or not lon_col:
    st.error("Could not find Latitude/Longitude columns in the bundled file.")
    st.stop()

df = df[pd.to_numeric(df[lat_col], errors="coerce").notna() &
        pd.to_numeric(df[lon_col], errors="coerce").notna()].copy()
if df.empty:
    st.error("No rows with numeric Latitude/Longitude to map.")
    st.stop()

df[lat_col] = df[lat_col].astype(float)
df[lon_col] = df[lon_col].astype(float)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("### Your location")
    mode = st.radio("Set location by:", ["Address search", "Coordinates (lat/lon)", "Click on map"], index=0)
    nearest_k = st.slider("How many nearest sites?", 1, 50, 10)

# ---------- Set location ----------
center_dataset = (df[lat_col].median(), df[lon_col].median())

if mode == "Address search":
    with st.form("addr_form", clear_on_submit=False):
        user_text = st.text_input("Enter address (e.g., 'Norwood SA')", value=st.session_state.geocoded_text or "")
        submitted = st.form_submit_button("Set location from address")
    if submitted and user_text.strip():
        r = geocode_cached(user_text.strip())
        if r:
            st.session_state.user_loc = (r[0], r[1])
            st.session_state.geocoded_text = r[2]
            st.success(f"Location set: {r[2]}")
        else:
            st.warning("Address not found.")
elif mode == "Coordinates (lat/lon)":
    with st.form("coord_form", clear_on_submit=False):
        lat_in = st.number_input("Latitude", value=st.session_state.user_loc[0] if st.session_state.user_loc else center_dataset[0], format="%.6f")
        lon_in = st.number_input("Longitude", value=st.session_state.user_loc[1] if st.session_state.user_loc else center_dataset[1], format="%.6f")
        submitted = st.form_submit_button("Set location from coordinates")
    if submitted:
        st.session_state.user_loc = (float(lat_in), float(lon_in))
        st.session_state.geocoded_text = f"{lat_in:.6f}, {lon_in:.6f}"
        st.success(f"Location set: {st.session_state.geocoded_text}")
else:  # Click on map
    st.info("Click the mini map below; then click 'Use clicked point' to set the location.")
    mini = folium.Map(location=list(center_dataset), zoom_start=11, control_scale=True)
    folium.LatLngPopup().add_to(mini)
    mret = st_folium(mini, height=300, width=None, key="picker_map")
    if mret and mret.get("last_clicked"):
        clicked = (mret["last_clicked"]["lat"], mret["last_clicked"]["lng"])
        st.write(f"Clicked: {clicked[0]:.6f}, {clicked[1]:.6f}")
        if st.button("Use clicked point"):
            st.session_state.user_loc = (float(clicked[0]), float(clicked[1]))
            st.session_state.geocoded_text = f"{clicked[0]:.6f}, {clicked[1]:.6f}"
            st.success("Location set from click.")

# Controls to clear / show current location
cols_top = st.columns([1, 3, 6])
with cols_top[0]:
    if st.button("Clear location"):
        st.session_state.user_loc = None
        st.session_state.geocoded_text = None
with cols_top[1]:
    st.caption(f"Current location: {st.session_state.geocoded_text or '— not set —'}")

# ---------- Compute nearest ----------
nearest_df = None
if st.session_state.user_loc:
    df = df.copy()
    df["distance_km"] = df.apply(lambda r: geodesic(st.session_state.user_loc, (r[lat_col], r[lon_col])).km, axis=1)
    nearest_df = df.sort_values("distance_km").head(nearest_k)

# ---------- Map ----------
if st.session_state.user_loc:
    center, zoom = st.session_state.user_loc, 13
else:
    center, zoom = center_dataset, 11

m = folium.Map(location=list(center), zoom_start=zoom, control_scale=True)
cluster = MarkerCluster(name="OFE / Off-road Sites").add_to(m)

# User marker + radius
if st.session_state.user_loc:
    folium.Marker(st.session_state.user_loc, tooltip="You are here", icon=folium.Icon(color="red")).add_to(m)
    folium.Circle(st.session_state.user_loc, radius=2000, color="#cc0000", fill=False).add_to(m)

# Plot points (nearest or all)
to_plot = nearest_df if nearest_df is not None else df
for _, row in to_plot.iterrows():
    title = build_name(row, cols_map)
    html = popup_html(row, cols_map)
    color = icon_color(row.get(cols_map["confirmed"]) if cols_map["confirmed"] else None)
    folium.Marker(
        [row[lat_col], row[lon_col]],
        popup=folium.Popup(html, max_width=360),
        tooltip=title,
        icon=folium.Icon(color=color, icon="ok-sign")
    ).add_to(cluster)

# Auto-fit bounds to user + nearest pins
if st.session_state.user_loc and nearest_df is not None and not nearest_df.empty:
    bounds = [[st.session_state.user_loc[0], st.session_state.user_loc[1]]] + nearest_df[[lat_col, lon_col]].values.tolist()
    m.fit_bounds(bounds, padding=(30, 30))

st_folium(m, height=580, width=None, key="main_map")

# ---------- Nearest list (styled, clickable name, features) ----------
if nearest_df is not None:
    st.markdown("### Nearest locations")

    park_col = cols_map.get("park") or cols_map.get("name")
    addr_col = cols_map.get("address")
    link_col = cols_map.get("link")
    feat_cols = cols_map.get("features", [])

    disp = nearest_df.copy()
    disp["Distance (km)"] = disp["distance_km"].map(lambda x: f"{x:.2f}")

    # Features summary
    def summarise_features(row):
        feats = [str(row[c]).strip() for c in feat_cols if pd.notna(row.get(c)) and str(row.get(c)).strip()]
        return ", ".join(feats) if feats else ""
    disp["Features"] = disp.apply(summarise_features, axis=1)

    # Clickable Park Name
    def clickable_name(row):
        name_val = str(row.get(park_col)) if park_col and pd.notna(row.get(park_col)) else "Outdoor Fitness Site"
        link_val = str(row.get(link_col)) if link_col and pd.notna(row.get(link_col)) else None
        if link_val and link_val.strip().startswith("http"):
            return f'<a href="{link_val.strip()}" target="_blank">{name_val}</a>'
        return name_val
    disp["Park Name"] = disp.apply(clickable_name, axis=1)

    cols_to_show = ["Park Name"]
    if addr_col: cols_to_show.append(addr_col)
    cols_to_show += ["Features", "Distance (km)"]
    rename_map = {addr_col: "Address"} if addr_col else {}

    df_display = disp[cols_to_show].rename(columns=rename_map)

    # Render clickable HTML table with clean styling
    html_table = df_display.to_html(escape=False, index=False, border=0)
    html_styled = f"""
    <style>
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 14px;
    }}
    th {{
      text-align: left;
      background-color: #f6f6f6;
      border-bottom: 2px solid #ddd;
      padding: 6px;
    }}
    td {{
      text-align: left;
      padding: 6px;
      border-bottom: 1px solid #eee;
      vertical-align: top;
    }}
    a {{
      color: #0072C6;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    </style>
    <div style="overflow-x:auto; border:1px solid #ccc; border-radius:6px; padding:8px; background:white;">
    {html_table}
    </div>
    """
    components.html(html_styled, height=420, scrolling=True)
else:
    st.info("Set a location (address, coordinates, or click) to see the nearest list.")
