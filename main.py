import argparse, os, re
from collections import defaultdict
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio

THEME = "plotly_white"
MAX_POINTS_PER_TRACE = 50_000

SENSORS_TO_PLOT = [
    # Oxidizer
    {"column": "PT-OX-04", "name": "PT-OX-04", "color": "#391FE0", "yaxis": "y1"},
    {"column": "PT-OX-02", "name": "PT-OX-02", "color": "#4199E1", "yaxis": "y1"},
    {"column": "PT-OX-201", "name": "PT-OX-201", "color": "#97C1E4", "yaxis": "y1"},
    {"column": "TC-OX-04", "name": "TC-OX-04", "color": "#2C6CCC", "yaxis": "y2"},
    {"column": "TC-OX-02", "name": "TC-OX-02", "color": "#4491AD", "yaxis": "y2"},
    {"column": "TC-OX-202", "name": "TC-OX-202", "color": "#9BC2DD", "yaxis": "y2"},
    {"column": "PI-OX-02", "name": "PI-OX-02", "color": "#9a28b3", "yaxis": "y4"},
    {"column": "PI-OX-03", "name": "PI-OX-03", "color": "#e662bc", "yaxis": "y4"},
    {"column": "RTD-OX", "name": "RTD-OX", "color": "#286CD1", "yaxis": "y5"},
    # Fuel
    {"column": "PT-FU-04", "name": "PT-FU-04", "color": "#80240B", "yaxis": "y1"},
    {"column": "PT-FU-02", "name": "PT-FU-02", "color": "#A3451D", "yaxis": "y1"},
    {"column": "PT-FU-201", "name": "PT-FU-201", "color": "#C77047", "yaxis": "y1"},
    {"column": "TC-FU-04", "name": "TC-FU-04", "color": "#D42828", "yaxis": "y2"},
    {"column": "TC-FU-02", "name": "TC-FU-02", "color": "#BE3C47", "yaxis": "y2"},
    {"column": "TC-FU-202", "name": "TC-FU-202", "color": "#B34C6C", "yaxis": "y2"},
    {"column": "PI-FU-02", "name": "PI-FU-02", "color": "#e32a33", "yaxis": "y4"},
    {"column": "PI-FU-03", "name": "PI-FU-03", "color": "#bd486f", "yaxis": "y4"},
    {"column": "RTD-FU", "name": "RTD-FU", "color": "#F70D0D", "yaxis": "y5"},
    # He
    {"column": "PT-HE-01", "name": "PT-HE-01", "color": "#2EB613", "yaxis": "y1"},
    {"column": "PT-HE-201", "name": "PT-HE-201", "color": "#87D197", "yaxis": "y1"},
    {"column": "TC-HE-201", "name": "TC-HE-201", "color": "#10842F", "yaxis": "y2"},
    # Other
    {"column": "FMS", "name": "FMS", "color": "#dada0a", "yaxis": "y3"},
]

X_AXIS_LABEL = "Time"
Y_AXIS_LABELS = {
    "y1": "Pressure [psia]",
    "y2": "Temperature [K]",
    "y3": "Mass [lbf]",
    "y4": "Position Indicator (0/1)",
    "y5": "RTD voltage [V]",
}

DEV5_TIME, DEV6_TIME = "Dev5_BCLS_ai_time", "Dev6_BCLS_ai_time"

DEV5_CHANNELS = [
    "PT-FU-04",
    "PT-HE-01",
    "PT-OX-02",
    "PT-N2-01",
    "PT-FU-02",
    "PT-OX-04",
    "TC-OX-04",
    "TC-FU-04",
    "TC-OX-02",
    "TC-FU-02",
    "FMS",
    "RTD-OX",
    "RTD-FU",
    "PT-FU-202",
    "PT-OX-202",
    "TC-HE-201",
]

DEV6_CHANNELS = ["TC-FU-BOTTOM", "TC-OX-202", "TC-FU-202", "TC-FU-UPPER", "PT-CHAMBER"]


def _direct_pairs(cols):
    pairs = defaultdict(list)
    for c in cols:
        if c.lower().endswith("_time"):
            base = re.sub(r"(_time|_TIME)$", "", c)
            if base in cols:
                pairs[c].append(base)
    return pairs


def _pi_pairs(cols):
    pairs = defaultdict(list)
    for c in cols:
        m = re.match(r"^BCLS_di_time_(.+)$", c)
        if m and m.group(1) in cols:
            pairs[c].append(m.group(1))
    return pairs


def _bcls_pairs(cols):
    pairs = defaultdict(list)
    if DEV5_TIME in cols:
        for ch in DEV5_CHANNELS:
            if ch in cols:
                pairs[DEV5_TIME].append(ch)
    if DEV6_TIME in cols:
        for ch in DEV6_CHANNELS:
            if ch in cols:
                pairs[DEV6_TIME].append(ch)
    return pairs


def _find_groups(cols):
    groups = defaultdict(list)
    for d in (_direct_pairs(cols), _pi_pairs(cols), _bcls_pairs(cols)):
        for t, ds in d.items():
            groups[t].extend(ds)
    return groups


def csv_to_parquet(input_csv: str) -> str:
    """Optimized CSV to Parquet conversion"""
    print("Reading CSV header...")
    header = pd.read_csv(input_csv, nrows=0)
    cols = list(header.columns)
    groups = _find_groups(cols)

    if not groups:
        raise ValueError("No valid time-column groupings found in CSV")

    usecols = {t for t in groups} | {d for ds in groups.values() for d in ds}
    print(
        f"Found {len(groups)} time column groups, {len(usecols)} total columns to process"
    )

    # Read entire CSV at once with optimizations
    print("Reading CSV data...")
    df = pd.read_csv(
        input_csv,
        usecols=list(usecols),
        low_memory=False,
        on_bad_lines="skip",
        engine="c",
    )

    print("Processing time groups...")
    all_frames = []

    for tcol, dcols in groups.items():
        if tcol not in df.columns:
            continue

        # Create subset with time column and data columns
        subset_cols = [tcol] + [c for c in dcols if c in df.columns]
        subset = df[subset_cols].copy()

        # Convert time column to datetime
        subset[tcol] = pd.to_datetime(subset[tcol], errors="coerce", utc=True)
        subset = subset.dropna(subset=[tcol])

        if subset.empty:
            continue

        # Set time as index
        subset = subset.set_index(tcol)

        # Handle duplicate indices BEFORE adding to all_frames
        if subset.index.duplicated().any():
            subset = subset.groupby(level=0).mean()

        # Convert all data columns to numeric at once
        for c in dcols:
            if c in subset.columns:
                subset[c] = pd.to_numeric(subset[c], errors="coerce")

        all_frames.append(subset)
        print(f"  Processed {tcol}: {len(subset)} rows, {len(subset.columns)} sensors")

    if not all_frames:
        raise ValueError("No valid data found after processing all groups")

    # Combine all frames with outer join
    print("Combining data frames...")
    combined = pd.concat(all_frames, axis=1, join="outer").sort_index()
    combined = combined.reset_index().rename(columns={"index": "timestamp"})

    # Save to parquet
    parquet_path = os.path.splitext(input_csv)[0] + ".parquet"
    print(f"Saving to {parquet_path}...")
    combined.to_parquet(parquet_path, index=False, engine="pyarrow")

    print(
        f"✓ Conversion complete: {len(combined)} rows, {len(combined.columns)} columns"
    )
    return parquet_path


def _thin(x, y, maxn):
    if maxn is None or len(y) <= maxn:
        return x.values, y.values
    idx = np.linspace(0, len(y) - 1, maxn, dtype=int)
    return x.values[idx], y.values[idx]


def _trace_axis_id(k):
    return "y" if k.lower() == "y1" else k.lower()


def _layout_axis_key(k):
    return "yaxis" if k.lower() == "y1" else f"yaxis{int(k[1:])}"


def plot_parquet(parquet_path: str, html_out: str, start: str | None, end: str | None):
    pio.templates.default = THEME
    print("Loading parquet file...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    if start or end:
        df = df.loc[start:end]

    print(f"Plotting data: {len(df)} rows, {len(df.columns)} columns")
    fig = go.Figure()
    used_axes = []
    traces_added = 0

    for s in SENSORS_TO_PLOT:
        c = s["column"]
        if c not in df.columns:
            continue

        y = pd.to_numeric(df[c], errors="coerce")
        mask = y.notna()

        if not mask.any():
            continue

        x_vals, y_vals = _thin(df.index[mask], y[mask], MAX_POINTS_PER_TRACE)
        yaxis_key = s.get("yaxis", "y1").lower()

        if yaxis_key not in used_axes:
            used_axes.append(yaxis_key)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=s.get("name", c),
                line=dict(color=s.get("color")),
                yaxis=_trace_axis_id(yaxis_key),
            )
        )
        traces_added += 1
        print(f"  Added trace: {s.get('name', c)} ({len(y_vals)} points)")

    if traces_added == 0:
        print("WARNING: No traces were added to the plot!")
        print(f"Available columns in data: {list(df.columns)}")
        print(f"Requested sensors: {[s['column'] for s in SENSORS_TO_PLOT]}")

    fig.update_layout(xaxis=dict(title=X_AXIS_LABEL), hovermode="x unified")
    used_axes.sort(key=lambda a: int(a[1:]) if a[1:].isdigit() else 1)

    step = 0.14 / max(1, len(used_axes) - 1) if len(used_axes) > 1 else 0
    for i, yk in enumerate(used_axes):
        k = _layout_axis_key(yk)
        t = Y_AXIS_LABELS.get(yk, yk)
        if yk == "y1":
            d = dict(title=dict(text=t), side="left", position=0.0, showgrid=True)
        else:
            side = "right" if i % 2 else "left"
            pos = (1 - (i // 2) * step) if side == "right" else ((i // 2 + 1) * step)
            pos = max(0.02, min(0.98, pos))
            d = dict(
                title=dict(text=t),
                overlaying="y",
                side=side,
                position=pos,
                showgrid=False,
            )
        fig.update_layout(**{k: d})

    print(f"Saving plot to {html_out}...")
    fig.write_html(html_out)
    print(f"✓ Plot saved with {traces_added} traces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    path = args.input_path
    base = os.path.splitext(os.path.basename(path))[0]

    if path.lower().endswith(".csv"):
        parquet_path = csv_to_parquet(path)
    elif path.lower().endswith((".parquet", ".pq")):
        parquet_path = path
    else:
        raise SystemExit("input must be .csv or .parquet")

    html_out = os.path.join("output", f"{base}.html")
    plot_parquet(parquet_path, html_out, args.start, args.end)
    print(f"\n✓ Complete! Plot saved to: {html_out}")


if __name__ == "__main__":
    main()
