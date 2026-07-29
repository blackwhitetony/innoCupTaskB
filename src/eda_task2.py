# AI辅助说明：Kimi K2.6，v2.6，Moonshot AI，2026-07-29，
# 生成/辅助内容：任务二探索性数据分析（EDA）可视化脚本。
# 该内容经作者核对修改后用于 B 题论文插图。

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import load_processed_data

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame):
    """图1：目标变量 home_charging_available 分布（柱状图+饼图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["home_charging_available"].value_counts().sort_index()
    labels = ["无家庭充电 (0)", "有家庭充电 (1)"]
    colors = ["#e74c3c", "#27ae60"]

    # 柱状图
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("家庭充电设施可用性分布", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("家庭充电状态")
    axes[0].set_ylabel("样本数")
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}\n({val / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # 饼图
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.02, 0.02),
    )
    axes[1].set_title("家庭充电设施占比", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_target_distribution.png", dpi=300, bbox_inches="tight")
    print(f"已保存: {OUTPUT_DIR / 'task2_target_distribution.png'}")
    plt.close(fig)


def plot_charging_distance_by_target(df: pd.DataFrame):
    """图2：最近公共充电站距离按家庭充电状态分组箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"无家庭充电": "#e74c3c", "有家庭充电": "#27ae60"}
    label_map = {0: "无家庭充电", 1: "有家庭充电"}
    df_plot = df.copy()
    df_plot["home_charging_label"] = df_plot["home_charging_available"].map(label_map)

    sns.boxplot(
        x="home_charging_label",
        y="nearest_charging_station_km",
        data=df_plot,
        palette=palette,
        ax=ax,
        linewidth=1.2,
        order=["无家庭充电", "有家庭充电"],
    )
    ax.set_title("最近公共充电站距离分布（按家庭充电状态）", fontsize=14, fontweight="bold")
    ax.set_xlabel("家庭充电状态")
    ax.set_ylabel("最近公共充电站距离 (km)")

    # 标注中位数
    for i, cls in enumerate([0, 1]):
        med = df.loc[df["home_charging_available"] == cls, "nearest_charging_station_km"].median()
        ax.annotate(
            f"中位数\n{med:.1f} km",
            xy=(i, med),
            fontsize=9,
            ha="center",
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_charging_distance_boxplot.png", dpi=300, bbox_inches="tight")
    print(f"已保存: {OUTPUT_DIR / 'task2_charging_distance_boxplot.png'}")
    plt.close(fig)


def plot_citytype_by_target(df: pd.DataFrame):
    """图3：城市类型与家庭充电可用性的交叉分析（堆叠柱状图）。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 构建交叉表（百分比）
    ct = pd.crosstab(df["city_type"], df["home_charging_available"], normalize="index") * 100
    city_labels = {0: "农村 (Rural)", 1: "郊区 (Suburban)", 2: "城市 (Urban)"}
    ct.index = ct.index.map(city_labels)

    ct.plot(
        kind="bar",
        stacked=True,
        color=["#e74c3c", "#27ae60"],
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("不同城市类型的家庭充电可用性比例", fontsize=14, fontweight="bold")
    ax.set_xlabel("城市类型")
    ax.set_ylabel("比例 (%)")
    ax.legend(["无家庭充电", "有家庭充电"], loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # 在柱上标注百分比
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_citytype_stacked_bar.png", dpi=300, bbox_inches="tight")
    print(f"已保存: {OUTPUT_DIR / 'task2_citytype_stacked_bar.png'}")
    plt.close(fig)


def main():
    df = load_processed_data()
    print(f"加载数据: {df.shape}")

    plot_target_distribution(df)
    plot_charging_distance_by_target(df)
    plot_citytype_by_target(df)

    print(f"\n所有任务二 EDA 图表已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
