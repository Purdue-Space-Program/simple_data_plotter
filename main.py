import argparse, os, re
from collections import defaultdict
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio
import random

THEME = "plotly_white"
MAX_POINTS_PER_TRACE = 50_000
# MAX_POINTS_PER_TRACE = 5_000


use_davids_auto_sensors = True

if use_davids_auto_sensors == False:

    SENSORS_TO_PLOT = [
        # Oxidizer
        {"column": "PT-OX-04", "name": "PT-OX-04", "color": "#391FE0", "yaxis": "y1"},
        {"column": "PT-OX-02", "name": "PT-OX-02", "color": "#4199E1", "yaxis": "y1"},
        {"column": "PT-OX-201", "name": "PT-OX-201", "color": "#97C1E4", "yaxis": "y1"},
        {"column": "PT-OX-202", "name": "PT-OX-202", "color": "#CFD7DE", "yaxis": "y1"},
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
        {"column": "PT-FU-202", "name": "PT-FU-202", "color": "#F6B090", "yaxis": "y1"},
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

    SENSORS_TO_PLOT_NAMES = [sensor["name"] for sensor in SENSORS_TO_PLOT]

else:

    SENSORS_TO_PLOT_NAMES = [
        "PT-OX-02",
        "PT-OX-04",
        "PT-OX-201",
        "PT-OX-202",
        
        "TC-OX-02",
        "TC-OX-04",
        "TC-OX-202",
        "TC-OX-201",
        "RTD-OX",
        
        "PI-OX-02",
        "PI-OX-03",
        
        
        "PT-FU-06",
        "PT-FU-04",
        "PT-FU-02",
        "PT-FU-201",
        "PT-FU-202",
        
        "TC-FU-04",
        "TC-FU-02",
        "TC-FU-202",
        "TC-FU-201",
        "RTD-FU", 
        
        "PI-FU-02",
        "PI-FU-03",
        "PI-FU-04"
        
        "PT-HE-01",
        "PT-HE-201",
        "TC-HE-201",
        
        "SV-N2-02_STATE",
        "SV-N2-02",
        
        "FMS",
    ]


    def FluidNameToColor(name: str) -> str:
        name_upper = name.upper()

        if "-OX" in name_upper:
            # sensor_color = "#3EABFF"
            # Random shade of blue
            r = random.randint(0, 100)
            g = random.randint(100, 200)
            b = random.randint(200, 255)
            sensor_color = f"#{r:02X}{g:02X}{b:02X}"
        elif "-FU" in name_upper:
            # sensor_color = "#6D0000"
            # Random shade of red
            r = random.randint(200, 255)
            g = random.randint(50, 150)
            b = random.randint(0, 100)
            sensor_color = f"#{r:02X}{g:02X}{b:02X}"
        elif "-HE" in name_upper:
            # sensor_color = "#6D0000"
            # Random shade of green
            r = random.randint(0, 100)
            g = random.randint(200, 255)
            b = random.randint(0, 100)
            sensor_color = f"#{r:02X}{g:02X}{b:02X}"
        elif "-N2" in name_upper:
            # Random shade of purple
            r = random.randint(200, 255)
            g = random.randint(0, 100)
            b = random.randint(200, 255)
            sensor_color = f"#{r:02X}{g:02X}{b:02X}"
        elif "-WA" in name_upper:
            # Random shade of blue
            r = random.randint(0, 100)
            g = random.randint(100, 200)
            b = random.randint(200, 255)
            sensor_color = f"#{r:02X}{g:02X}{b:02X}"
        elif "FMS" in name_upper:
            sensor_color = "#DCEB0E"
        else:
            sensor_color = "#000000"
        
        return(sensor_color)

    def SensorTypeToAxis(name: str) -> str:
        name_upper = name.upper()

        if "PT" in name_upper:
            sensor_axis = "y1"
        elif "PI-" in name_upper:
            sensor_axis = "y2"
        elif "TC-" in name_upper:
            sensor_axis = "y3"
        elif "RTD-" in name_upper:
            sensor_axis = "y4"
        elif "FMS" in name_upper:
            sensor_axis = "y5"
        else:
            sensor_axis = "y6"

        return(sensor_axis)

    SENSORS_TO_PLOT = []

    for sensor_name in SENSORS_TO_PLOT_NAMES:
        sensor_color = FluidNameToColor(sensor_name)
        sensor_axis = SensorTypeToAxis(sensor_name)
        SENSORS_TO_PLOT.append({"column": sensor_name, "name": sensor_name, "color": sensor_color, "yaxis": sensor_axis},)


X_AXIS_LABEL = "Time [H:M:S:milliseconds]"
Y_AXIS_LABELS = {
    "y1": "Pressure [psia]",
    "y2": "Position Indicator [0/1]",
    "y3": "Fuel Temperature [°K]",
    "y4": "RTD Voltage (V)",
    "y5": "Mass (lbf)",
    "y6": "unknown sensor",
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

DEV6_CHANNELS = [
    "PV-FU-04", 
    "SV-N2-02", 
    "SV-N2-03", 
    "TC-FU-202", 
    "TC-OX-202", 
    "TC-FU-VENT", 
    "PT-FU-06" 
    "PT-CHAMBER", 
    "TC-BATTERY", 
    "HS_CAMERA", 
]



channels = [
    [DEV5_CHANNELS, DEV5_TIME], 
    [DEV6_CHANNELS, DEV6_TIME]
    ]


def DirectPairs(csv_columns):
    print("\nDirectPairs")
    
    pairs = {}
    
    for csv_column in csv_columns:        
        
        if csv_column.lower().endswith("_time"):
            sensor_name = re.sub(r"(_time|_TIME)$", "", csv_column)
            
            for channel, time in channels:
                # check if it is time
                # if (csv_column in (c[1] for c in channels)):
                #     pairs[time] = [sensor_name]

                # elif sensor_name in channel:
                if sensor_name in SENSORS_TO_PLOT_NAMES:
                    pairs[csv_column] = [sensor_name]

                elif channel == channels[-1][0]:
                    print(f"Warning: Column '{sensor_name}' not found; skipping.")


    # ##### compare with old version
    # print (f"\n{pairs}")
    # pairs = None

    # pairs = defaultdict(list)
    # for c in csv_columns:
    #     if c.lower().endswith("_time"):
    #         base = re.sub(r"(_time|_TIME)$", "", c)
    #         if base in csv_columns:
    #             pairs[c].append(base)

    # print (f"\n{pairs}\n")
    # ##### compare with old version
    
    return pairs


def MakePIPairs(csv_columns):
    print("\nMakePIPairs")
    
    pairs = {}
    
    for csv_column in csv_columns:
        m = re.match(r"^BCLS_di_time_(.+)$", csv_column)
        
        if m != None:
            sensor_name = m.group(1)
            if sensor_name in SENSORS_TO_PLOT_NAMES:
            
                if (m and sensor_name in csv_columns):

                        pairs[csv_column] = [m.group(1)]
                        
                        
            else:
                print(f"Warning: Column '{csv_column}' not found; skipping.")


    # ######### compare with old version
    # print(f"\n{pairs}")
    # pairs = None        
        
    # pairs = defaultdict(list)
    
    # for csv_column in csv_columns:
    #     # if csv_column == "BCLS_di_time_PI-OX-02":
    #     #     pass
    #     m = re.match(r"^BCLS_di_time_(.+)$", csv_column)
    #     if m and m.group(1) in csv_columns:
    #         pairs[csv_column].append(m.group(1))
            

    # print(f"\n{pairs}\n")
    # ######### compare with old version
    
    return pairs


def BCLSPairs(csv_columns):
    print("\nBCLSPairs")
    
    pairs = defaultdict(list) # dictionary that automatically creates list whenever new key is attempted

    for channel, time in channels:
        if time in csv_columns:
            for sensor_name in channel:
                
                if sensor_name in csv_columns:
                    if (sensor_name in SENSORS_TO_PLOT_NAMES):
                        pairs[time].append(sensor_name)

                    elif channel == channels[-1][0]:
                        print(f"Warning: Column '{sensor_name}' not found; skipping.")




    # ############ compare with old version
    # print(f"\n{pairs}")
    # pairs = None                
            
    # pairs = defaultdict(list)
    
    # if DEV5_TIME in csv_columns:
    #     for ch in DEV5_CHANNELS:
    #         if ch in csv_columns:
    #             pairs[DEV5_TIME].append(ch)
    # if DEV6_TIME in csv_columns:
    #     for ch in DEV6_CHANNELS:
    #         if ch in csv_columns:
    #             pairs[DEV6_TIME].append(ch)

    # print(f"\n{pairs}\n")
    # ############ compare with old version

    return(pairs)


def FindGroups(csv_columns):
    groups = defaultdict(list)
    for d in (DirectPairs(csv_columns), MakePIPairs(csv_columns), BCLSPairs(csv_columns)):
        for t, ds in d.items():
            groups[t].extend(ds)
    return groups


def ConvertCSVToParquet(input_csv: str) -> str:
    """Optimized CSV to Parquet conversion"""
    print("Reading CSV header...")
    header = pd.read_csv(input_csv, nrows=0)
    csv_columns = list(header.columns)
    groups = FindGroups(csv_columns)

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
        on_bad_lines="warn",
        engine="c",
    )

    print("Processing time groups...")
    all_frames = []

    for time_column, data_columns in groups.items():
        if time_column not in df.columns:
            continue

        # Create subset with time column and data columns
        subset_cols = [time_column] + [c for c in data_columns if c in df.columns]
        subset = df[subset_cols].copy()

        # Convert time column to datetime
        subset[time_column] = pd.to_datetime(subset[time_column], errors="coerce", utc=True)
        subset = subset.dropna(subset=[time_column])

        if subset.empty:
            continue

        # Set time as index
        subset = subset.set_index(time_column)

        # Handle duplicate indices BEFORE adding to all_frames
        if subset.index.duplicated().any():
            subset = subset.groupby(level=0).mean()

        # Convert all data columns to numeric at once
        for c in data_columns:
            if c in subset.columns:
                subset[c] = pd.to_numeric(subset[c], errors="coerce")

        all_frames.append(subset)
        print(f"  Processed {time_column}: {len(subset)} rows, {len(subset.columns)} sensors")

    if not all_frames:
        raise ValueError("No valid data found after processing all groups")

    # Combine all frames with outer join
    print("Combining data frames...")
    combined = pd.concat(all_frames, axis=1, join="outer").sort_index()
    combined = combined.reset_index().rename(columns={"index": "timestamp"})

    # Save to parquet
    base = os.path.splitext(input_csv)[0]
    parquet_path = f"{base}.parquet"
    
    print(f"Saving to {parquet_path}...")
    combined.to_parquet(parquet_path, index=False, engine="pyarrow")

    print(
        f"✓ Conversion complete: {len(combined)} rows, {len(combined.columns)} columns"
    )
    return parquet_path


def _thin(x, y, maxn):
    if maxn is None:
        raise ValueError
    
    if len(y) <= maxn:
        return x.values, y.values
    idx = np.linspace(0, len(y) - 1, maxn, dtype=int)
    return x.values[idx], y.values[idx]


def _trace_axis_id(k):
    return "y" if k.lower() == "y1" else k.lower()


def _layout_axis_key(k):
    return "yaxis" if k.lower() == "y1" else f"yaxis{int(k[1:])}"


# def AssignLegendGroup(sensor_name):
#     """
#     Convert a sensor name like 'PT-OX-04' → 'OX_PT'.
#     If it doesn't match the pattern, return 'OTHER_MISC'.
#     """
#     m = re.match(r"([A-Z]+)-([A-Z]+)", sensor_name)
#     if m:
#         sensor_type, system = m.groups()
#         legend_group_name = f"{system}_{sensor_type}"
#     else:
#         legend_group_name = "OTHER_MISC"
#     return legend_group_name


    # def ChangeVisibility(fig, fluid_abbreviation, current_visibility, should_be_visible):
    #     """Return a list of True/False for each trace depending on whether its name contains system."""
    #     new_visibility = []
    #     for index, trace in enumerate(fig.data):
    #         name = trace.name or ""
    #         if fluid_abbreviation in name:
    #             new_visibility.append(should_be_visible)
    #         else:
    #             new_visibility.append(current_visibility[index])

    #     return new_visibility    



def PlotParquet(parquet_path: str, html_out: str, start: str | None, end: str | None):
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

    for sensor in SENSORS_TO_PLOT:
        column = sensor["column"]
        if column not in df.columns:
            continue

        y = pd.to_numeric(df[column], errors="coerce")
        mask = y.notna()

        if not mask.any():
            continue

        x_vals, y_vals = _thin(df.index[mask], y[mask], MAX_POINTS_PER_TRACE)
        yaxis_key = sensor.get("yaxis", "y1").lower()

        if yaxis_key not in used_axes:
            used_axes.append(yaxis_key)



        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=sensor.get("name", column),
                line=dict(color=sensor.get("color")),
                yaxis=_trace_axis_id(yaxis_key),
     
            )
        )
        traces_added += 1
        print(f"  Added trace: {sensor.get('name', column)} ({len(y_vals)} points)")

    
    
        
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Toggle OX",
                        method="update",
                        args=[{}, {}],
                        name="btn_ox",
                    ),
                    dict(
                        label="Toggle FU",
                        method="update",
                        args=[{}, {}],
                        name="btn_fu",
                    ),
                    dict(
                        label="Toggle HE",
                        method="update",
                        args=[{}, {}],
                        name="btn_he",
                    ),
                    dict(
                        label="Toggle PT",
                        method="update",
                        args=[{}, {}],
                        name="btn_pt",
                    ),
                    dict(
                        label="Toggle TC",
                        method="update",
                        args=[{}, {}],
                        name="btn_tc",
                    ),
                    dict(
                        label="Toggle RTD",
                        method="update",
                        args=[{}, {}],
                        name="btn_rtd",
                    ),
                    dict(
                        label="Toggle PI",
                        method="update",
                        args=[{}, {}],
                        name="btn_pi",
                    ),
                    dict(
                        label="Toggle FMS",
                        method="update",
                        args=[{}, {}],
                        name="btn_fms",
                    ),
                    dict(
                        label="Show ALL",
                        method="update",
                        args=[{}, {}],
                        name="btn_all",
                    ),
                ],
                direction="down",
                pad=dict(r=10, t=10),
                bgcolor="#333",
                font=dict(color="white"),
            )
        ]
    )




    def export_plot_with_dynamic_buttons(fig, path, div_id="my_fig"):
        """Export Plotly HTML with JS that adds dynamic group toggling."""

        # Step 1 — save HTML normally
        fig.write_html(path, 
                       include_plotlyjs="cdn", 
                       full_html=True, 
                       div_id=div_id)


        # Step 2 — JavaScript code for real-time group toggle
        js_code = """
                <script>
                (function(){
                    function toggleGroup(tag) {
                    const gd = document.getElementById("my_fig");
                    if (!gd) return;

                    const data = gd.data;

                    // Determine which traces belong to this group
                    const groupIdx = [];
                    for (let i = 0; i < data.length; i++) {
                        if (data[i].name && data[i].name.includes(tag)) {
                            groupIdx.push(i);
                        }
                    }

                    if (tag === "ALL") {
                        // Show everything
                        Plotly.restyle(gd, {visible: true});
                        return;
                    }

                    // Get current group visibility (treat undefined as visible)
                    const groupVis = groupIdx.map(i =>
                        (data[i].visible === undefined || data[i].visible)
                    );

                    // If ANY are visible → turn ALL off
                    // If ALL are hidden → turn ALL on
                    const target = groupVis.some(v => v) ? false : true;

                    // Build final visibility array
                    let vis = data.map(t => (t.visible === undefined || t.visible));

                    for (const idx of groupIdx) {
                        vis[idx] = target;
                    }

                    Plotly.restyle(gd, {visible: vis});
                    }


                    // Helper: find a descendant element whose trimmed textContent equals label
                    function findElementByExactText(root, label) {
                        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
                        let node;
                        while (node = walker.nextNode()) {
                            const txt = (node.textContent || "").trim();
                            if (txt === label) return node;
                        }
                        return null;
                    }

                    // Attach handlers once the buttons exist. Use MutationObserver to detect rendering.
                    function attachWhenReady() {
                    const gd = document.getElementById("my_fig");
                    if (!gd) {
                        console.warn("toggleGroup: plot div not found (#my_fig)");
                        return;
                    }

                    const labels = [
                        "Toggle OX",
                        "Toggle FU",
                        "Toggle HE",
                        "Toggle PT",
                        "Toggle TC",
                        "Toggle RTD",
                        "Toggle PI",
                        "Toggle FMS",
                        "Show ALL",
                    ];

                    const handlers = [
                        () => toggleGroup("-OX"),
                        () => toggleGroup("-FU"),
                        () => toggleGroup("-HE"),
                        () => toggleGroup("PT-"),
                        () => toggleGroup("TC-"),
                        () => toggleGroup("RTD-"),
                        () => toggleGroup("PI-"),
                        () => toggleGroup("FMS"),
                        () => toggleGroup("ALL"),
                    ];

                    let attached = 0;
                    const timeoutAt = Date.now() + 3000; // try up to 3s

                    const tryAttach = () => {
                        for (let i = 0; i < labels.length; i++) {
                            const label = labels[i];
                            const el = findElementByExactText(gd, label);
                            if (el && !el.__toggle_attached) {
                                el.addEventListener("click", handlers[i]);
                                el.__toggle_attached = true;
                                attached++;
                                console.log("toggleGroup: attached handler to button:", label, el);
                            }
                        }
                        if (attached === labels.length) {
                            console.log("toggleGroup: all handlers attached");
                            observer.disconnect();
                            return;
                        }
                        if (Date.now() > timeoutAt) {
                            console.warn("toggleGroup: timed out waiting for buttons; attached:", attached);
                            observer.disconnect();
                        }
                    };

                    // Try immediately in case elements are already present
                    tryAttach();

                    // Observe DOM changes to catch the buttons when Plotly renders them
                    const observer = new MutationObserver(() => {
                        tryAttach();
                    });
                    observer.observe(gd, { childList: true, subtree: true });
                }

                if (document.readyState === "complete" || document.readyState === "interactive") {
                    setTimeout(attachWhenReady, 50);
                } else {
                    window.addEventListener("DOMContentLoaded", () => setTimeout(attachWhenReady, 50));
                }
                })();
                </script>
                """



        theme_toggle_js = """
        <script>
        (function() {
            const btn = document.createElement("input");
            btn.type = "checkbox";
            btn.id = "themeToggle";
            btn.style.position = "fixed";
            btn.style.top = "10px";
            btn.style.left = "10px";
            btn.style.zIndex = "9999";
            btn.title = "Toggle Dark Mode";

            const lbl = document.createElement("label");
            lbl.htmlFor = "themeToggle";
            lbl.innerText = "🌞/🌙";
            lbl.style.position = "fixed";
            lbl.style.top = "12px";
            lbl.style.left = "40px";
            lbl.style.color = "black";
            lbl.style.fontFamily = "sans-serif";
            lbl.style.fontSize = "25px";
            lbl.style.cursor = "pointer";
            lbl.style.zIndex = "9999";

            document.body.appendChild(lbl);
            document.body.appendChild(btn);

            btn.addEventListener("change", () => {
                const gd = document.getElementById("my_fig");
                if (!gd) return;

                const isDark = btn.checked;
                const newTemplate = isDark ? "plotly_dark" : "plotly_white";

                const layoutUpdate = {
                    template: newTemplate,
                    paper_bgcolor: isDark ? "#111" : "#fff",
                    plot_bgcolor: isDark ? "#111" : "#fff",
                    font: { color: isDark ? "#eee" : "#000" },
                };

                console.log("Switching theme to:", newTemplate);
                Plotly.react(gd, gd.data, { ...gd.layout, ...layoutUpdate });

                // Change page background and label color too
                document.body.style.backgroundColor = layoutUpdate.paper_bgcolor;
                lbl.style.color = layoutUpdate.font.color;
            });
        })();
        </script>
        """






        # Step 3 — append JS before </body>
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        html = html.replace("</body>", js_code + theme_toggle_js + "\n</body>")

        # Step 4 — write modified HTML back
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)




    # fig.update_layout(
        
    #     current_visibility = [trace.visible for trace in fig.data]

    #     updatemenus=[
    #         dict(
    #             type="buttons",
    #             showactive=False,
    #             buttons=[
    #                 dict(
    #                     label="Toggle OX",
    #                     method="restyle",
    #                     args=[{"visible": ChangeVisibility(fig, "OX", current_visibility, True)}],
    #                     args2=[{"visible": ChangeVisibility(fig, "OX", current_visibility, False)}],
    #                 )
    #             ],
    #             bgcolor="#0071ae",
    #             bordercolor="#676767",
    #             font=dict(color="white"),
    #             pad=dict(t=10, r=10),
    #             x=1.15,
    #             y=1,
    #         ),

    #         dict(
    #             type="buttons",
    #             showactive=False,
    #             buttons=[
    #                 dict(
    #                     label="Toggle FU",
    #                     method="restyle",
    #                     args=[{"visible": fu_visibilities}],
    #                     args2=[{"visible": fu_visibilities_inv}],
    #                 )
    #             ],
    #             bgcolor="#900000",
    #             bordercolor="#676767",
    #             font=dict(color="white"),
    #             pad=dict(t=10, r=10),
    #             x=1.30,
    #             y=1,
    #         ),
    #     ]
    # )

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
    export_plot_with_dynamic_buttons(fig, html_out, div_id="my_fig")
    print(f"✓ Plot saved with {traces_added} traces")


def main():
    
    DEFAULT_PATH = "data/reduced_11-9-Hotfire-Attempts_new.parquet"
    _SENTINEL = object()

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "input_path",
        nargs="?",
        default=_SENTINEL,
    )

    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)

    args = ap.parse_args()


    if args.input_path is _SENTINEL:
        # user did NOT provide input_path
        print(f"WARNING!!!!!!!!!!! No input file provided, using default input path: {DEFAULT_PATH}")
        args.input_path = DEFAULT_PATH

    path_to_input_file = args.input_path
    input_file_name = os.path.splitext(os.path.basename(path_to_input_file))[0]

    if path_to_input_file.lower().endswith(".csv"):
        parquet_path = ConvertCSVToParquet(path_to_input_file)
    elif path_to_input_file.lower().endswith((".parquet", ".pq")):
        parquet_path = path_to_input_file
    else:
        raise SystemExit("input must be .csv or .parquet")

    html_out = os.path.join("output", f"{input_file_name}.html")
    PlotParquet(parquet_path, html_out, args.start, args.end)
    print(f"\n✓ Complete! Plot saved to: {html_out}")


if __name__ == "__main__":
    main()
