import argparse, os, re
from collections import defaultdict
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio

THEME = "plotly_white"
MAX_POINTS_PER_TRACE = 50_000

SENSORS_TO_PLOT = [
    {"column": "PT-OX-02", "name": "PT-OX-02", "color": "#DFD91F", "yaxis": "y1"},
    {"column": "TC-OX-02", "name": "TC-OX-02", "color": "#A4CC10", "yaxis": "y2"},
    {"column": "PT-OX-04", "name": "PT-OX-04", "color": "#FF7FD2", "yaxis": "y1"},
    {"column": "PT-OX-201", "name": "PT-OX-201", "color": "#820857", "yaxis": "y1"},
    {"column": "PI-OX-02", "name": "PI-OX-02", "color": "#1f77b4", "yaxis": "y4"},
    {"column": "PI-OX-03", "name": "PI-OX-03", "color": "#ff7f0e", "yaxis": "y4"},
    {"column": "FMS", "name": "FMS", "color": "#9467bd", "yaxis": "y3"},
]

X_AXIS_LABEL = "Time"
Y_AXIS_LABELS = {
    "y1": "Pressure (PSI)",
    "y2": "Temperature (K)",
    "y3": "Mass (lbs)",
    "y4": "Position Indicator (0/1)",
}

DEV5_TIME, DEV6_TIME = "Dev5_BCLS_ai_time", "Dev6_BCLS_ai_time"
DEV5_CHANNELS = ["FMS", "PT-FU-04", "PT-OX-04", "TC-FU-04", "TC-HE-201"]
DEV6_CHANNELS = ["TC-FU-202", "TC-OX-202", "TC-FU-VENT"]


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
    CHUNKSIZE = 1_000_000
    header = pd.read_csv(input_csv, nrows=0)
    cols = list(header.columns)
    groups = _find_groups(cols)
    usecols = {t for t in groups} | {d for ds in groups.values() for d in ds}
    chunks = {t: [] for t in groups}

    with pd.read_csv(
        input_csv,
        chunksize=CHUNKSIZE,
        low_memory=False,
        usecols=list(usecols),
        on_bad_lines="skip",
    ) as rdr:
        for chunk in rdr:
            for tcol, dcols in groups.items():
                subset = chunk[[c for c in [tcol] + dcols if c in chunk]].copy()
                subset.dropna(subset=[tcol], inplace=True)
                if subset.empty:
                    continue
                subset[tcol] = pd.to_datetime(subset[tcol], errors="coerce", utc=True)
                subset.dropna(subset=[tcol], inplace=True)
                subset.set_index(tcol, inplace=True)
                for c in dcols:
                    subset[c] = pd.to_numeric(subset[c], errors="coerce")
                chunks[tcol].append(subset[dcols])

    frames = [pd.concat(v).sort_index() for v in chunks.values() if v]
    df = pd.concat(frames, axis=1).sort_index()
    df = df.reset_index().rename(columns={"index": "timestamp"})

    parquet_path = os.path.splitext(input_csv)[0] + ".parquet"
    df.to_parquet(parquet_path, index=False)
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
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df = df.loc[start:end] if (start or end) else df
    fig = go.Figure()
    used_axes = []

    for s in SENSORS_TO_PLOT:
        c = s["column"]
        if c not in df:
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

    fig.update_layout(xaxis=dict(title=X_AXIS_LABEL), hovermode="x unified")
    used_axes.sort(key=lambda a: int(a[1:]) if a[1:].isdigit() else 1)
    offset_total, step = 0.14, 0.14 / max(1, len(used_axes) - 1)
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

    fig.write_html(html_out)


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
    print(f"saved plot → {html_out}")


if __name__ == "__main__":
    main()
