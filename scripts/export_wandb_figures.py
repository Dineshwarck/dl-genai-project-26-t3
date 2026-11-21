"""
Export Training Visualizations from Weights & Biases (W&B)
Project: SurfaceGuard AI (24F2004781-t32026)
Entity: dineshwar-developer-indian-institute-of-technology

Directly pulls logged metrics for all 3 architectures from W&B GraphQL API
and generates publication-quality SVG and PNG figures in reports/figures/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import httpx

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENTITY = "dineshwar-developer-indian-institute-of-technology"
PROJECT = "24F2004781-t32026"

# Load WANDB_API_KEY from environment or .env
api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    for env_path in [".env", os.path.expanduser("~/.env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        clean_line = line.strip().lstrip("\ufeff")
                        if clean_line.startswith("WANDB_API_KEY="):
                            api_key = clean_line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                        elif len(clean_line) >= 30 and "=" not in clean_line and not clean_line.startswith("#"):
                            api_key = clean_line.strip().strip('"').strip("'")
                            break
                if api_key:
                    break
            except Exception:
                pass

if not api_key:
    print("Error: Could not locate WANDB_API_KEY in .env.")
    sys.exit(1)

print("=" * 70)
print("  SURFACEGUARD AI: EXPORTING W&B TRAINING VISUALIZATIONS")
print(f"  Target: {ENTITY}/{PROJECT}")
print(f"  Destination: {OUTPUT_DIR}")
print("=" * 70)

client = httpx.Client(auth=("api", api_key), timeout=30.0)

# Verified representative run IDs for each architecture
RUN_TARGETS = {
    "Model 2: ResAttentionUNet (DDP)": {
        "id": "j8lagz9s",
        "color": "#16a34a", # Green
        "keys": {
            "epoch": "resattunet/epoch",
            "train_loss": "resattunet/train_loss",
            "val_loss": "resattunet/val_loss",
            "fg_dice": "resattunet/foreground_dice",
            "score": "resattunet/official_score",
            "fp_score": "resattunet/false_positive_score"
        }
    },
    "Model 3: PretrainedResNet34 (2-Phase)": {
        "id": "b2dt23w8",
        "color": "#dc2626", # Red
        "keys": {
            "epoch": "epoch",
            "train_loss": "train/loss",
            "val_loss": "val/loss",
            "fg_dice": "val/foreground_dice",
            "score": "val/official_score",
            "fp_score": "val/false_positive_score"
        }
    },
    "Model 1: ScratchUNet (Baseline)": {
        "id": "2q8cuec0",
        "color": "#2563eb", # Blue
        "keys": {
            "epoch": "epoch",
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "fg_dice": "foreground_dice",
            "score": "official_score",
            "fp_score": "false_positive_score"
        }
    }
}

query = """
query GetRunHistory($entity: String!, $project: String!, $run: String!) {
    project(name: $project, entityName: $entity) {
        run(name: $run) {
            displayName
            history(samples: 100)
        }
    }
}
"""

extracted_data = {}

for model_name, cfg in RUN_TARGETS.items():
    run_id = cfg["id"]
    print(f"\n[+] Fetching metrics for '{model_name}' (Run ID: {run_id})...")
    res = client.post(
        "https://api.wandb.ai/graphql",
        json={"query": query, "variables": {"entity": ENTITY, "project": PROJECT, "run": run_id}}
    )
    if res.status_code != 200:
        print(f"  -> Error fetching run {run_id}: {res.status_code}")
        continue
    
    r_data = res.json().get("data", {}).get("project", {}).get("run", {})
    hist_raw = r_data.get("history", [])
    print(f"  -> Retrieved {len(hist_raw)} logged step records from W&B.")
    
    records = []
    for line in hist_raw:
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    
    if records:
        df = pd.DataFrame(records)
        extracted_data[model_name] = {
            "df": df,
            "cfg": cfg
        }

if not extracted_data:
    print("[Error] No data extracted from W&B.")
    sys.exit(1)

# -------------------------------------------------------------
# Function to generate clean SVG chart
# -------------------------------------------------------------
def create_svg_chart(title, subtitle, x_label, y_label, series_list, out_path, y_min=None, y_max=None):
    width, height = 980, 520
    margin_l, margin_r = 90, 260
    margin_t, margin_b = 75, 75
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_x = [x for s in series_list for x in s["x"]]
    all_y = [y for s in series_list for y in s["y"]]

    x_min = min(all_x) if all_x else 1
    x_max = max(all_x) if all_x else 40
    y_min_val = y_min if y_min is not None else (min(all_y) * 0.95 if all_y else 0.0)
    y_max_val = y_max if y_max is not None else (max(all_y) * 1.05 if all_y else 1.0)

    def to_pt(x, y):
        cx = margin_l + ((x - x_min) / (x_max - x_min if x_max > x_min else 1)) * plot_w
        cy = margin_t + plot_h - ((y - y_min_val) / (y_max_val - y_min_val if y_max_val > y_min_val else 1)) * plot_h
        return cx, cy

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">')
    svg.append(f'<rect width="{width}" height="{height}" fill="#ffffff" />')
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" rx="4" />')

    # Title & Subtitle
    svg.append(f'<text x="{margin_l + plot_w/2}" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#0f172a">{title}</text>')
    svg.append(f'<text x="{margin_l + plot_w/2}" y="52" text-anchor="middle" font-size="12" fill="#64748b">{subtitle}</text>')

    # Y Gridlines
    n_yticks = 6
    for i in range(n_yticks + 1):
        y_val = y_min_val + i * (y_max_val - y_min_val) / n_yticks
        _, cy = to_pt(x_min, y_val)
        svg.append(f'<line x1="{margin_l}" y1="{cy}" x2="{margin_l + plot_w}" y2="{cy}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3" />')
        svg.append(f'<text x="{margin_l - 12}" y="{cy + 4}" text-anchor="end" font-size="11" font-weight="500" fill="#64748b">{y_val:.2f}</text>')

    # X Gridlines
    n_xticks = 8
    for i in range(n_xticks + 1):
        x_val = int(x_min + i * (x_max - x_min) / n_xticks)
        cx, _ = to_pt(x_val, y_min_val)
        svg.append(f'<line x1="{cx}" y1="{margin_t}" x2="{cx}" y2="{margin_t + plot_h}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3" />')
        svg.append(f'<text x="{cx}" y="{margin_t + plot_h + 22}" text-anchor="middle" font-size="11" font-weight="500" fill="#64748b">Epoch {x_val}</text>')

    # Box border
    svg.append(f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cbd5e1" stroke-width="1.5" />')

    # Axis Labels
    svg.append(f'<text x="{margin_l + plot_w/2}" y="{height - 20}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{x_label}</text>')
    svg.append(f'<text transform="rotate(-90)" x="-{margin_t + plot_h/2}" y="28" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{y_label}</text>')

    # Plot Lines
    leg_y = margin_t + 16
    for s in series_list:
        coords = [to_pt(x, y) for x, y in zip(s["x"], s["y"])]
        pts = " ".join([f"{cx:.1f},{cy:.1f}" for cx, cy in coords])
        dash = 'stroke-dasharray="6,4"' if s.get("dash") else ''
        svg.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{s.get("width", 2.6)}" {dash} points="{pts}" />')

        # Marker dots
        step = max(1, len(coords) // 8)
        for idx in range(0, len(coords), step):
            px, py = coords[idx]
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.5" />')

        # Legend entry
        svg.append(f'<rect x="{width - margin_r + 16}" y="{leg_y - 12}" width="{margin_r - 24}" height="24" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />')
        svg.append(f'<line x1="{width - margin_r + 24}" y1="{leg_y}" x2="{width - margin_r + 54}" y2="{leg_y}" stroke="{s["color"]}" stroke-width="2.6" {dash} />')
        svg.append(f'<text x="{width - margin_r + 62}" y="{leg_y + 4}" font-size="11" font-weight="500" fill="#1e293b">{s["name"]}</text>')
        leg_y += 30

    svg.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[OK] Generated: {out_path}")

# -------------------------------------------------------------
# Chart 1: Training & Validation Loss Convergence
# -------------------------------------------------------------
loss_series = []
for model_name, item in extracted_data.items():
    df, cfg = item["df"], item["cfg"]
    color = cfg["color"]
    k_epoch = cfg["keys"]["epoch"]
    k_train = cfg["keys"]["train_loss"]
    k_val = cfg["keys"]["val_loss"]

    if k_train in df.columns:
        sub = df[[k_epoch, k_train]].dropna()
        loss_series.append({
            "name": f"{model_name.split(':')[1].split('(')[0].strip()} (Train)",
            "x": [int(x) for x in sub[k_epoch].values],
            "y": [float(y) for y in sub[k_train].values],
            "color": color,
            "dash": False,
            "width": 2.5
        })

    if k_val in df.columns:
        sub = df[[k_epoch, k_val]].dropna()
        loss_series.append({
            "name": f"{model_name.split(':')[1].split('(')[0].strip()} (Val)",
            "x": [int(x) for x in sub[k_epoch].values],
            "y": [float(y) for y in sub[k_val].values],
            "color": color,
            "dash": True,
            "width": 2.0
        })

loss_svg = os.path.join(OUTPUT_DIR, "wandb_loss_convergence.svg")
create_svg_chart(
    title="W&B Training & Validation Loss Convergence Across 40 Epochs",
    subtitle=f"Weights & Biases Project: {ENTITY}/{PROJECT} (3 Multi-Label Architectures)",
    x_label="Training Epoch",
    y_label="Multi-Task ComboLoss (Focal + Dice)",
    series_list=loss_series,
    out_path=loss_svg,
    y_min=0.40,
    y_max=1.30
)

# -------------------------------------------------------------
# Chart 2: Official Composite Score Progression
# -------------------------------------------------------------
score_series = []
for model_name, item in extracted_data.items():
    df, cfg = item["df"], item["cfg"]
    color = cfg["color"]
    k_epoch = cfg["keys"]["epoch"]
    k_score = cfg["keys"]["score"]

    if k_score in df.columns:
        sub = df[[k_epoch, k_score]].dropna()
        score_series.append({
            "name": f"{model_name.split(':')[1].strip()}",
            "x": [int(x) for x in sub[k_epoch].values],
            "y": [float(y) for y in sub[k_score].values],
            "color": color,
            "dash": False,
            "width": 2.8
        })

score_svg = os.path.join(OUTPUT_DIR, "wandb_official_score_progression.svg")
create_svg_chart(
    title="W&B Official Competition Score Progression Over Epochs",
    subtitle="Metric Formula: 0.70 * Foreground Dice + 0.20 * FP Silence + 0.10 * Empty Accuracy",
    x_label="Training Epoch",
    y_label="Official Score (0.0 to 1.0)",
    series_list=score_series,
    out_path=score_svg,
    y_min=0.25,
    y_max=0.75
)

# -------------------------------------------------------------
# Chart 3: Foreground Dice Overlap Progression
# -------------------------------------------------------------
dice_series = []
for model_name, item in extracted_data.items():
    df, cfg = item["df"], item["cfg"]
    color = cfg["color"]
    k_epoch = cfg["keys"]["epoch"]
    k_dice = cfg["keys"]["fg_dice"]

    if k_dice in df.columns:
        sub = df[[k_epoch, k_dice]].dropna()
        dice_series.append({
            "name": f"{model_name.split(':')[1].strip()}",
            "x": [int(x) for x in sub[k_epoch].values],
            "y": [float(y) for y in sub[k_dice].values],
            "color": color,
            "dash": False,
            "width": 2.8
        })

dice_svg = os.path.join(OUTPUT_DIR, "wandb_foreground_dice_progression.svg")
create_svg_chart(
    title="W&B Validation Foreground Dice Progression",
    subtitle="Mean Foreground Dice on Ground-Truth Defect Pixels Across Classes",
    x_label="Training Epoch",
    y_label="Foreground Dice Score (0.0 to 1.0)",
    series_list=dice_series,
    out_path=dice_svg,
    y_min=0.10,
    y_max=0.68
)

print("\n" + "=" * 70)
print(f"  EXPORT COMPLETED SUCCESSFULLY! CHARTS SAVED IN: {OUTPUT_DIR}/")
print("=" * 70)
