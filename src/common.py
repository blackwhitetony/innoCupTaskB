# AI辅助说明：DeepSeek-V4-Pro，v4，DeepSeek，2026-07-29，
# 生成/辅助内容：任务一公共数据加载、分层划分、评估指标函数。
# 该内容经作者核对修改后用于 B 题任务一训练流水线。

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

RANDOM_STATE = 2026
TEST_SIZE = 0.2
CATBOOST_TASK_TYPE = os.environ.get("CATBOOST_TASK", "CPU")  # "GPU" or "CPU"


def load_processed_data() -> pd.DataFrame:
    """加载 processed_data.xlsx（UTF-8，已清洗）。"""
    root = Path(__file__).resolve().parent.parent
    path = root / "processed_data.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    对完整数据按目标变量分层划分 80/20，固定随机种子 2026。

    Returns
    -------
    X_train, X_valid, y_train, y_valid, le
    """
    if drop_cols is None:
        drop_cols = []

    y_raw = df[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X = df.drop(columns=[target_col] + drop_cols)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_valid, y_train, y_valid, le


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    prefix: str = "",
) -> dict:
    """计算并打印分类评估指标。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        f"{prefix}accuracy": acc,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}balanced_accuracy": balanced_acc,
    }

    print(f"\n{'='*40}")
    print(f"{prefix} 评估结果".strip())
    print(f"{'='*40}")
    print(f"准确率 (Accuracy): {acc:.5f}")
    print(f"宏平均 F1 (Macro F1): {macro_f1:.5f}")
    print(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.5f}")
    print("\n混淆矩阵:")
    print(pd.DataFrame(cm, index=le.classes_.astype(str), columns=le.classes_.astype(str)))
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=le.classes_.astype(str)))
    return metrics


def save_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    le: LabelEncoder,
    filename: str = "task1_results.csv",
):
    """保存验证集预测结果 CSV（行号 + 真实标签 + 预测标签 + 各类概率）。"""
    root = Path(__file__).resolve().parent.parent / "outputs"
    root.mkdir(exist_ok=True)
    out = pd.DataFrame({"y_true": le.inverse_transform(y_true), "y_pred": le.inverse_transform(y_pred)})
    if y_proba is not None:
        for i, cls in enumerate(le.classes_):
            out[f"proba_{str(cls)}"] = y_proba[:, i]
    out.to_csv(root / filename, index=False, encoding="utf-8-sig")
    print(f"结果已保存至: {root / filename}")
