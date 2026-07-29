# AI辅助说明：Kimi K2.6，v2.6，Moonshot AI，2026-07-29，
# 生成/辅助内容：任务三 EDA 可视化（目标分布、原始评分分布、充电困难度箱线图、EV知识得分箱线图）。
# 该内容经作者核对修改后用于 B 题任务三建模。

import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import load_processed_data

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
sns.set(style="whitegrid", font="SimHei")

# 解决中文显示问题（尝试常见中文字体）
for font in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]:
    try:
        plt.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def plot_target_distribution(df: pd.DataFrame, out_dir: Path):
    """任务三标签分布。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["task3_anxiety_label"].value_counts().sort_index()
    colors = ["#3498db", "#e74c3c"]
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="black")
    ax.set_xlabel("里程焦虑标签")
    ax.set_ylabel("样本数")
    ax.set_title("任务三：里程焦虑敏感度标签分布")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (≤5)", "1 (>5)"])
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "task3_target_distribution.png", dpi=300)
    plt.close(fig)
    print(f"已保存: {out_dir / 'task3_target_distribution.png'}")


def plot_score_distribution(df: pd.DataFrame, out_dir: Path):
    """原始 range_anxiety_score 分布及阈值线。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = df["range_anxiety_score"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color="steelblue", edgecolor="black")
    ax.axvline(x=5.0, color="red", linestyle="--", linewidth=2, label="阈值 = 5")
    ax.set_xlabel("Range Anxiety Score (原始)")
    ax.set_ylabel("样本数")
    ax.set_title("任务三：原始里程焦虑评分分布及二分类阈值")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "task3_score_distribution.png", dpi=300)
    plt.close(fig)
    print(f"已保存: {out_dir / 'task3_score_distribution.png'}")


def plot_box_by_target(df: pd.DataFrame, out_dir: Path):
    """按标签绘制充电困难度和 EV 知识得分的箱线图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 充电困难度近似值
    df_temp = df.copy()
    df_temp["charging_difficulty"] = df_temp["nearest_charging_station_km"] / (
        df_temp["charging_station_accessibility"] + 1.0
    )
    sns.boxplot(
        x="task3_anxiety_label", y="charging_difficulty", data=df_temp,
        ax=axes[0], palette=["#3498db", "#e74c3c"]
    )
    axes[0].set_title("充电困难度按焦虑标签分布")
    axes[0].set_xlabel("里程焦虑标签")
    axes[0].set_ylabel("充电困难度 (km / (accessibility + 1))")
    axes[0].set_xticklabels(["0 (≤5)", "1 (>5)"])

    # EV 知识得分
    sns.boxplot(
        x="task3_anxiety_label", y="ev_knowledge_score", data=df,
        ax=axes[1], palette=["#3498db", "#e74c3c"]
    )
    axes[1].set_title("EV 知识得分按焦虑标签分布")
    axes[1].set_xlabel("里程焦虑标签")
    axes[1].set_ylabel("EV Knowledge Score")
    axes[1].set_xticklabels(["0 (≤5)", "1 (>5)"])

    fig.tight_layout()
    fig.savefig(out_dir / "task3_boxplot_by_target.png", dpi=300)
    plt.close(fig)
    print(f"已保存: {out_dir / 'task3_boxplot_by_target.png'}")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path):
    """核心特征与目标的相关性热力图。"""
    core_cols = [
        "task3_anxiety_label",
        "daily_commute_km",
        "weekly_travel_distance_km",
        "nearest_charging_station_km",
        "charging_station_accessibility",
        "home_charging_available",
        "ev_knowledge_score",
        "battery_replacement_concern",
        "technology_affinity_score",
        "fuel_expense_per_month",
        "monthly_charging_cost",
        "age",
        "annual_income",
    ]
    corr = df[core_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("任务三：核心特征相关性热力图")
    fig.tight_layout()
    fig.savefig(out_dir / "task3_correlation_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"已保存: {out_dir / 'task3_correlation_heatmap.png'}")


def main():
    df = load_processed_data()
    print(f"原始数据: {df.shape}")

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_target_distribution(df, out_dir)
    plot_score_distribution(df, out_dir)
    plot_box_by_target(df, out_dir)
    plot_correlation_heatmap(df, out_dir)

    print("\n任务三 EDA 完成")


if __name__ == "__main__":
    main()
