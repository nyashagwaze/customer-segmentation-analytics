"""Prepare README-friendly screenshots from generated figure outputs."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    figures_dir = root / "outputs" / "figures"
    screenshots_dir = root / "assets" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "spend_distribution.png": "01_spend_distribution.png",
        "avg_spend_by_region.png": "02_avg_spend_by_region.png",
        "correlation_heatmap.png": "03_correlation_heatmap.png",
        "kmeans_silhouette_by_k.png": "04_kmeans_silhouette_by_k.png",
        "cluster_profile_heatmap.png": "05_cluster_profile_heatmap.png",
    }

    copied = []
    missing = []

    for src_name, dst_name in mapping.items():
        src = figures_dir / src_name
        dst = screenshots_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((src_name, dst_name))
        else:
            missing.append(src_name)

    print("Screenshots directory:", screenshots_dir)
    if copied:
        print("Copied:")
        for src_name, dst_name in copied:
            print(f"  - {src_name} -> assets/screenshots/{dst_name}")
    if missing:
        print("Missing source figures:")
        for src_name in missing:
            print(f"  - {src_name}")


if __name__ == "__main__":
    main()
