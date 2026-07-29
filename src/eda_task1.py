# AI辅助说明：DeepSeek-V4-Pro，v4，DeepSeek，2026-07-29，
# 生成/辅助内容：任务一探索性数据分析（EDA）可视化脚本。
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
    """图1：目标变量 ev_adoption_likelihood 分布（柱状图+饼图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 柱状图
    order = ["Low", "Medium", "High"]
    counts = df["ev_adoption_likelihood"].value_counts().reindex(order)
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("电动汽车采用意愿分布", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("采用意愿等级")
    axes[0].set_ylabel("样本数")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                     f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=11)

    # 饼图
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, explode=(0.02, 0.02, 0.02))
    axes[1].set_title("采用意愿占比", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task1_target_distribution.png", dpi=300, bbox_inches="tight")
    print(f"已保存: {OUTPUT_DIR / 'task1_target_distribution.png'}")
    plt.close(fig)


def plot_income_by_target(df: pd.DataFrame):
    """图2：年收入按采用意愿等级的箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    order = ["Low", "Medium", "High"]
    palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#27ae60"}

    sns.boxplot(x="ev_adoption_likelihood", y="annual_income", data=df,
                order=order, palette=palette, ax=ax, linewidth=1.2)
    ax.set_title("不同采用意愿群体的年收入分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("采用意愿等级")
    ax.set_ylabel("年收入 (元)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/10000:.0f}万"))

    # 标注中位数
    for i, cls in enumerate(order):
        med = df.loc[df["ev_adoption_likelihood"] == cls, "annual_income"].median()
        ax.annotate(f"中位数\n{med/10000:.1f}万", xy=(i, med), fontsize=9, ha="center",
                    color="white", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task1_income_boxplot.png", dpi=300, bbox_inches="tight")
    print(f"已保存: {OUTPUT_DIR / 'task1_income_boxplot.png'}")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    """图3：数值特征相关性热力图（精简版，选 15 个核心特征）。"""
    core_cols = [
        "age", "annual_income", "daily_commute_km", "weekly_travel_distance_km",
        "vehicle_age_years", "fuel_expense_per_month", "charging_station_accessibility",
        "nearest_charging_station_km", "environmental_awareness_score",
        "government_incentive_awareness", "technology_affinity_score",
        "range_anxiety_score", "battery_replacement_concern", "ev_knowledge_score",
        "monthly_charging_cost",
    ]
    corr = df[core_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 只保留下三角
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("核心数值特征相关性热力图", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task1_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"已保存: {OUTPUT_DIR / 'task1_correlation_heatmap.png'}")
    plt.close(fig)


def main():
    df = load_processed_data()
    print(f"加载数据: {df.shape}")

    plot_target_distribution(df)
    plot_income_by_target(df)
    plot_correlation_heatmap(df)

    print(f"\n所有 EDA 图表已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
