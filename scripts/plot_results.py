#!/usr/bin/env python3
"""Plot GEMM benchmark results with interactive HTML and PNG fallback."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bootstrap_local_venv():
    repo_root = Path(__file__).resolve().parents[1]
    venv_root = repo_root / "env"
    if not venv_root.exists():
        return
    for site_dir in venv_root.glob("lib/python*/site-packages"):
        if str(site_dir) not in sys.path:
            sys.path.insert(0, str(site_dir))


try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    bootstrap_local_venv()
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - import error is explicit runtime feedback
        raise SystemExit(
            "plotly not available. Install it in env/ or rerun with env python."
        ) from exc


DIM_ORDER = [32, 64, 128, 256, 512, 1024, 2048, 4096]
ARCH_ORDER = ["h100", "rtx5070", "unknown"]
DTYPE_ORDER = ["BF16", "FP32", "FP32-TC"]

COLOR_MAP = {
    "BF16 custom": "#1f77b4",
    "BF16 cuBLAS 32F": "#ff7f0e",
    "BF16 cuBLAS TC": "#2ca02c",
    "FP32 custom": "#d62728",
    "FP32 SGEMM": "#9467bd",
    "FP32 CUDA": "#8c564b",
    "FP32 pedantic": "#e377c2",
    "FP32 TF32": "#7f7f7f",
    "FP32-TC custom": "#bcbd22",
    "FP32-TC cuBLAS TF32": "#17becf",
}

DTYPE_MARKER = {
    "BF16": "circle",
    "FP32": "square",
    "FP32-TC": "diamond",
}

MATPLOTLIB_DTYPE_MARKER = {
    "BF16": "o",
    "FP32": "s",
    "FP32-TC": "D",
}

ARCH_DASH = {
    "h100": "solid",
    "rtx5070": "dash",
    "unknown": "dot",
}

SERIES_BY_DTYPE = {
    "BF16": [
        ("custom_ms", "custom", "custom"),
        ("cublas_ms", "cuBLAS 32F", "ref"),
        ("cublas_tc_ms", "cuBLAS TC", "ref"),
    ],
    "FP32": [
        ("custom_ms", "custom", "custom"),
        ("pedantic_ms", "cuBLAS pedantic", "ref"),
    ],
    "FP32-TC": [
        ("custom_ms", "custom", "custom"),
        ("tc_ms", "cuBLAS TF32"),
    ],
}


def safe_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def infer_arch(csv_path: Path, row: dict) -> str:
    target = (row.get("target_arch") or "").strip().lower()
    if target:
        return target
    parent = csv_path.parent.name.strip().lower()
    if parent in {"h100", "rtx5070"}:
        return parent
    return "unknown"


def load_results(csv_paths):
    rows = []
    for csv_path in csv_paths:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dtype = (row.get("dtype") or "").strip()
                if dtype not in SERIES_BY_DTYPE:
                    continue
                rows.append(
                    {
                        "csv_path": csv_path,
                        "dim": int(row["dim"]),
                        "dtype": dtype,
                        "arch": infer_arch(csv_path, row),
                        "gpu_name": (row.get("gpu_name") or "unknown").strip(),
                        "sm": (row.get("sm") or "").strip(),
                        "variant": (row.get("variant") or "").strip(),
                        "desc": (row.get("desc") or "").strip(),
                        "custom_ms": safe_float(row.get("custom_ms")),
                        "cublas_32f_ms": safe_float(row.get("cublas_32f_ms")),
                        "cublas_tc_ms": safe_float(row.get("cublas_tc_ms")),
                        "sgemm_ms": safe_float(row.get("sgemm_ms")),
                        "cuda_ms": safe_float(row.get("cuda_ms")),
                        "pedantic_ms": safe_float(row.get("pedantic_ms")),
                        "tc_ms": safe_float(row.get("tc_ms")),
                        "l2_error": safe_float(row.get("l2_error")),
                    }
                )
    return rows


def find_csvs(results_root: Path):
    csvs = []
    arch_csvs = sorted(results_root.glob("*/benchmark_results.csv"))
    if arch_csvs:
        csvs.extend(arch_csvs)
    else:
        root_csv = results_root / "benchmark_results.csv"
        if root_csv.exists():
            csvs.append(root_csv)
    return csvs


def gflops(dim: int, time_ms: float) -> float:
    if time_ms <= 0:
        return 0.0
    return 2.0 * (dim ** 3) / (time_ms * 1e6)


def value_for_metric(row: dict, field: str, metric: str) -> float:
    if metric == "time":
        return safe_float(row.get(field))
    return gflops(row["dim"], safe_float(row.get(field)))


def series_label(dtype: str, label: str) -> str:
    return f"{dtype} {label}"


def trace_kind_and_label(dtype: str, series_entry):
    if len(series_entry) == 3:
        field, label, kind = series_entry
        return field, label, kind
    field, label = series_entry
    kind = "ref" if "cuBLAS" in label or label == "pedantic" else "custom"
    return field, label, kind


def build_series(rows, metric: str):
    traces = []
    seen_arches = []
    seen_dtypes = []

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arch"], row["dtype"])].append(row)
        if row["arch"] not in seen_arches:
            seen_arches.append(row["arch"])
        if row["dtype"] not in seen_dtypes:
            seen_dtypes.append(row["dtype"])

    seen_arches.sort(key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else len(ARCH_ORDER))
    seen_dtypes.sort(key=lambda d: DTYPE_ORDER.index(d) if d in DTYPE_ORDER else len(DTYPE_ORDER))

    for arch in seen_arches:
        for dtype in seen_dtypes:
            key = (arch, dtype)
            if key not in grouped:
                continue
            entries = sorted(grouped[key], key=lambda r: r["dim"])
            for series_entry in SERIES_BY_DTYPE[dtype]:
                field, label, kind = trace_kind_and_label(dtype, series_entry)
                xs = []
                ys = []
                hover = []
                for row in entries:
                    val = value_for_metric(row, field, metric)
                    if val <= 0:
                        continue
                    xs.append(row["dim"])
                    ys.append(val)
                    if kind == "custom":
                        hover.append((row["gpu_name"], row["sm"], row["variant"], row["desc"]))
                    else:
                        hover.append((row["gpu_name"], row["sm"], label))
                if not xs:
                    continue
                series = series_label(dtype, label)
                is_reference = kind == "ref"
                marker_symbol = DTYPE_MARKER.get(dtype, "circle")
                traces.append(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=f"{arch} | {series}",
                        legendgroup=series,
                        legendrank=100 if is_reference else 10,
                        line={
                            "color": "#006400" if is_reference else COLOR_MAP.get(series, "#444444"),
                            "dash": "dot" if is_reference else ARCH_DASH.get(arch, "dot"),
                            "width": 2.8,
                        },
                        marker={
                            "size": 9,
                            "symbol": marker_symbol,
                            "color": "#006400" if is_reference else COLOR_MAP.get(series, "#444444"),
                            "line": {"width": 1, "color": "#006400" if is_reference else COLOR_MAP.get(series, "#444444")},
                        },
                        meta={"arch": arch, "dtype": dtype},
                        customdata=hover,
                        hovertemplate=(
                            "arch=%{meta.arch}<br>"
                            "dtype=%{meta.dtype}<br>"
                            "N=%{x}<br>"
                            + ("time=%{y:.4f} ms" if metric == "time" else "GFLOPS=%{y:.2f}")
                            + (
                                "<br>gpu=%{customdata[0]}<br>sm=%{customdata[1]}<br>variant=%{customdata[2]}<br>desc=%{customdata[3]}"
                                if kind == "custom"
                                else "<br>ref=%{customdata[2]}<br>gpu=%{customdata[0]}<br>sm=%{customdata[1]}"
                            )
                            + "<extra></extra>"
                        ),
                    )
                )
    return traces, seen_arches, seen_dtypes


def build_bar_traces(rows, metric: str):
    traces = []
    grouped = defaultdict(list)
    seen_arches = []
    seen_dtypes = []
    for row in rows:
        grouped[row["arch"]].append(row)
        if row["arch"] not in seen_arches:
            seen_arches.append(row["arch"])
        if row["dtype"] not in seen_dtypes:
            seen_dtypes.append(row["dtype"])

    seen_arches.sort(key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else len(ARCH_ORDER))
    seen_dtypes.sort(key=lambda d: DTYPE_ORDER.index(d) if d in DTYPE_ORDER else len(DTYPE_ORDER))

    for arch in seen_arches:
        entries = sorted(grouped[arch], key=lambda r: r["dim"])
        for dtype in seen_dtypes:
            dtype_entries = [row for row in entries if row["dtype"] == dtype]
            if not dtype_entries:
                continue
            for series_entry in SERIES_BY_DTYPE[dtype]:
                field, label, kind = trace_kind_and_label(dtype, series_entry)
                xs = []
                ys = []
                hover = []
                for row in dtype_entries:
                    val = value_for_metric(row, field, metric)
                    if val <= 0:
                        continue
                    xs.append(str(row["dim"]))
                    ys.append(val)
                    if kind == "custom":
                        hover.append((row["gpu_name"], row["sm"], row["variant"], row["desc"]))
                    else:
                        hover.append((row["gpu_name"], row["sm"], label))
                if not xs:
                    continue
                series = series_label(dtype, label)
                is_reference = kind == "ref"
                traces.append(
                    go.Bar(
                        x=xs,
                        y=ys,
                        name=f"{arch} | {series}",
                        legendgroup=series,
                        legendrank=100 if is_reference else 10,
                        offsetgroup=f"{arch}-{series}",
                        marker={
                            "color": "#006400" if is_reference else COLOR_MAP.get(series, "#444444"),
                            "line": {"color": "#006400" if is_reference else COLOR_MAP.get(series, "#444444"), "width": 1},
                            "pattern": {
                                "shape": "/" if is_reference else "",
                                "fgcolor": "#ffffff",
                                "size": 8,
                            },
                        },
                        width=0.22,
                        customdata=hover,
                        hovertemplate=(
                            "arch=%{name}<br>"
                            "dtype=" + dtype + "<br>"
                            "N=%{x}<br>"
                            + ("time=%{y:.4f} ms")
                            + (
                                "<br>gpu=%{customdata[0]}<br>sm=%{customdata[1]}<br>variant=%{customdata[2]}<br>desc=%{customdata[3]}"
                                if kind == "custom"
                                else "<br>ref=%{customdata[2]}<br>gpu=%{customdata[0]}<br>sm=%{customdata[1]}"
                            )
                            + "<extra></extra>"
                        ),
                    )
                )
    return traces, seen_arches, seen_dtypes


def make_plotly_figure(rows, metric: str, title: str, y_label: str, default_log_x: bool, default_log_y: bool):
    traces, seen_arches, seen_dtypes = build_series(rows, metric)
    fig = go.Figure(data=traces)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        width=1280,
        height=780,
        font={"family": "DejaVu Sans, Arial, sans-serif", "size": 14},
        legend={
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "traceorder": "normal",
            "bordercolor": "rgba(0,0,0,0.25)",
            "borderwidth": 1,
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
        xaxis={"type": "log" if default_log_x else "linear"},
        yaxis={"type": "log" if default_log_y else "linear"},
    )

    xaxis = {
        "title": "Matrix size (N)",
        "tickmode": "array",
        "tickvals": DIM_ORDER,
        "ticks": "outside",
        "ticklen": 6,
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.15)",
        "zeroline": False,
    }
    yaxis = {
        "title": y_label,
        "ticks": "outside",
        "ticklen": 6,
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.15)",
        "zeroline": False,
    }
    fig.update_xaxes(**xaxis)
    fig.update_yaxes(**yaxis)

    return fig, seen_arches, seen_dtypes


def wrap_interactive_html(fig, div_id: str, title: str, seen_arches, seen_dtypes,
                          default_log_x: bool, default_log_y: bool, allow_log_x: bool = True):
    fig_html = pio.to_html(fig, include_plotlyjs="inline", full_html=False, div_id=div_id)
    arch_opts = ["all"] + seen_arches
    dtype_opts = ["all"] + seen_dtypes

    def options_html(values):
        return "".join(f'<option value="{escape(v)}">{escape(v)}</option>' for v in values)

    controls = f"""
<div style="max-width:1280px;margin:0 auto 12px auto;font-family:DejaVu Sans, Arial, sans-serif;font-size:14px;">
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:8px 12px;border:1px solid rgba(0,0,0,0.12);border-radius:8px;background:#fafafa;">
    <label><input id="{div_id}-logx" type="checkbox" {'checked' if default_log_x else ''} {'disabled' if not allow_log_x else ''} /> Log X</label>
    <label><input id="{div_id}-logy" type="checkbox" {'checked' if default_log_y else ''} /> Log Y</label>
    <label>Arch
      <select id="{div_id}-arch" style="margin-left:8px;padding:4px 8px;">
        {options_html(arch_opts)}
      </select>
    </label>
    <label>Dtype
      <select id="{div_id}-dtype" style="margin-left:8px;padding:4px 8px;">
        {options_html(dtype_opts)}
      </select>
    </label>
    <span style="color:#555;">{escape(title)}</span>
  </div>
</div>
"""

    script = f"""
<script>
(function() {{
  const gd = document.getElementById('{div_id}');
  const allowLogX = {str(allow_log_x).lower()};
  const logX = document.getElementById('{div_id}-logx');
  const logY = document.getElementById('{div_id}-logy');
  const archSel = document.getElementById('{div_id}-arch');
  const dtypeSel = document.getElementById('{div_id}-dtype');

  function applyFilter() {{
    const arch = archSel.value;
    const dtype = dtypeSel.value;
    const visible = gd.data.map(trace => {{
      const meta = trace.meta || {{}};
      const archOk = arch === 'all' || meta.arch === arch;
      const dtypeOk = dtype === 'all' || meta.dtype === dtype;
      return archOk && dtypeOk;
    }});
    Plotly.restyle(gd, {{visible: visible}});
    const relayout = {{
      'yaxis.type': logY && logY.checked ? 'log' : 'linear'
    }};
    if (allowLogX) {{
      relayout['xaxis.type'] = logX && logX.checked ? 'log' : 'linear';
    }}
    Plotly.relayout(gd, relayout);
  }}

  function tintCublasLegend() {{
    const texts = gd.querySelectorAll('.legend text');
    texts.forEach(t => {{
      const text = (t.textContent || '').toLowerCase();
      if (text.includes('cublas') || text.includes('pedantic')) {{
        t.setAttribute('fill', '#006400');
        t.setAttribute('font-weight', '700');
      }}
    }});
  }}

  archSel.addEventListener('change', applyFilter);
  dtypeSel.addEventListener('change', applyFilter);
  if (logX) logX.addEventListener('change', applyFilter);
  if (logY) logY.addEventListener('change', applyFilter);
  window.addEventListener('load', applyFilter);
  setTimeout(() => {{ applyFilter(); tintCublasLegend(); }}, 0);
  gd.on('plotly_afterplot', tintCublasLegend);
}})();
</script>
"""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; padding: 18px; background: #fff; }}
    .plot-wrap {{ max-width: 1280px; margin: 0 auto; }}
  </style>
</head>
<body>
  <div class="plot-wrap">
    {controls}
    {fig_html}
  </div>
  {script}
</body>
</html>
"""


def save_png(rows, metric: str, title: str, y_label: str, log_scale: bool, output_path: Path):
    traces, seen_arches, seen_dtypes = build_series(rows, metric)
    if not traces:
        return

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(13, 8))
    linestyle_by_arch = {"h100": "-", "rtx5070": "--", "unknown": ":"}
    for trace in traces:
        arch, series = trace.name.split(" | ", 1)
        dtype = series.split(" ", 1)[0]
        color = COLOR_MAP.get(series, "#444444")
        marker = MATPLOTLIB_DTYPE_MARKER.get(dtype, "o")
        linestyle = linestyle_by_arch.get(arch, "-")
        label = trace.name
        x = np.asarray(trace.x, dtype=float)
        y = np.asarray(trace.y, dtype=float)
        if log_scale:
            ax.loglog(
                x,
                y,
                marker=marker,
                linewidth=2.2,
                markersize=6,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        else:
            ax.plot(
                x,
                y,
                marker=marker,
                linewidth=2.2,
                markersize=6,
                label=label,
                color=color,
                linestyle=linestyle,
            )

    ax.set_title(title)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel(y_label)
    ax.set_xticks(DIM_ORDER)
    ax.set_xticklabels([str(d) for d in DIM_ORDER])
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, borderaxespad=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_bar_figure(rows, title: str, y_label: str, default_log_y: bool):
    traces, seen_arches, seen_dtypes = build_bar_traces(rows, metric="time")
    fig = go.Figure(data=traces)
    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        width=1280,
        height=780,
        font={"family": "DejaVu Sans, Arial, sans-serif", "size": 14},
        barmode="group",
        legend={
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "traceorder": "normal",
            "bordercolor": "rgba(0,0,0,0.25)",
            "borderwidth": 1,
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
        yaxis={"type": "log" if default_log_y else "linear"},
    )
    fig.update_xaxes(
        title="Matrix size (N)",
        type="category",
        categoryorder="array",
        categoryarray=[str(d) for d in DIM_ORDER],
        ticks="outside",
        ticklen=6,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False,
    )
    fig.update_yaxes(
        title=y_label,
        ticks="outside",
        ticklen=6,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False,
    )
    return fig, seen_arches, seen_dtypes


def save_bar_png(rows, title: str, y_label: str, output_path: Path, log_y: bool):
    traces, seen_arches, seen_dtypes = build_bar_traces(rows, metric="time")
    if not traces:
        return
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(len(DIM_ORDER))
    bar_count = max(len(traces), 1)
    width = 0.8 / bar_count
    for idx, trace in enumerate(traces):
        arch, series = trace.name.split(" | ", 1)
        color = getattr(trace.marker, "color", "#444444")
        vals = {int(xv): yv for xv, yv in zip(trace.x, trace.y)}
        y = [vals.get(dim, np.nan) for dim in DIM_ORDER]
        pos = x - 0.4 + idx * width + width / 2
        ax.bar(
            pos,
            y,
            width=width,
            label=trace.name,
            color=color,
            edgecolor=color,
            hatch="//" if "cuBLAS" in series else None,
            alpha=0.95,
        )

    ax.set_title(title)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DIM_ORDER])
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, borderaxespad=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GEMM benchmark results")
    parser.add_argument("--results-root", default="results", help="Root directory with benchmark CSVs")
    parser.add_argument("--output-dir", default="results/plots", help="Output directory for plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_csvs(results_root)
    if not csv_paths:
        raise SystemExit(f"No benchmark CSV found under {results_root}")

    rows = load_results(csv_paths)
    if not rows:
        raise SystemExit(f"No usable rows found in {len(csv_paths)} CSV file(s)")

    archs = sorted({r["arch"] for r in rows}, key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else len(ARCH_ORDER))
    dtypes = sorted({r["dtype"] for r in rows}, key=lambda d: DTYPE_ORDER.index(d) if d in DTYPE_ORDER else len(DTYPE_ORDER))

    print(f"Loaded {len(rows)} rows from {len(csv_paths)} CSV file(s)")
    print(f"Available arch: {', '.join(archs)}")
    print(f"Available dtype: {', '.join(dtypes)}")

    time_fig, seen_arches, seen_dtypes = make_plotly_figure(
        rows,
        metric="time",
        title="GEMM Runtime Comparison",
        y_label="Time (ms)",
        default_log_x=True,
        default_log_y=True,
    )
    time_html = wrap_interactive_html(
        time_fig,
        div_id="time-loglog",
        title="GEMM Runtime Comparison",
        seen_arches=seen_arches,
        seen_dtypes=seen_dtypes,
        default_log_x=True,
        default_log_y=True,
    )
    (output_dir / "time_loglog.html").write_text(time_html, encoding="utf-8")
    save_png(
        rows,
        metric="time",
        title="GEMM Runtime Comparison",
        y_label="Time (ms)",
        log_scale=True,
        output_path=output_dir / "time_loglog.png",
    )

    gflops_fig, seen_arches, seen_dtypes = make_plotly_figure(
        rows,
        metric="gflops",
        title="GEMM Throughput Comparison",
        y_label="GFLOPS",
        default_log_x=True,
        default_log_y=True,
    )
    gflops_html = wrap_interactive_html(
        gflops_fig,
        div_id="gflops-linear",
        title="GEMM Throughput Comparison",
        seen_arches=seen_arches,
        seen_dtypes=seen_dtypes,
        default_log_x=True,
        default_log_y=True,
    )
    (output_dir / "gflops_linear.html").write_text(gflops_html, encoding="utf-8")
    save_png(
        rows,
        metric="gflops",
        title="GEMM Throughput Comparison",
        y_label="GFLOPS",
        log_scale=True,
        output_path=output_dir / "gflops_linear.png",
    )

    bar_title = "GEMM Runtime Bar Comparison"
    bar_fig, seen_arches, seen_dtypes = make_bar_figure(rows, title=bar_title, y_label="Time (ms)", default_log_y=True)
    bar_html = wrap_interactive_html(
        bar_fig,
        div_id="time-bar",
        title=bar_title,
        seen_arches=seen_arches,
        seen_dtypes=seen_dtypes,
        default_log_x=False,
        default_log_y=True,
        allow_log_x=False,
    )
    (output_dir / "time_bar.html").write_text(bar_html, encoding="utf-8")
    save_bar_png(
        rows,
        title=bar_title,
        y_label="Time (ms)",
        output_path=output_dir / "time_bar.png",
        log_y=True,
    )

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
