# predict.py — 模型调用入口
# 使用方式：python predict.py
# 或：uv run python predict.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from catboost import CatBoostClassifier

from common import load_processed_data
from features import add_task1_features, add_task2_features, add_task3_features

MODEL_DIR = ROOT / "models"

# ---- 任务一：采用意愿三分类 ----
def predict_task1(df: pd.DataFrame):
    """返回 pred (array of str: Low/Medium/High) 和 proba (n×3)。"""
    df = add_task1_features(df)
    drop = ["ev_adoption_likelihood", "target_ord"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "task1_catboost.cbm"))
    return model.predict(X), model.predict_proba(X)

# ---- 任务二：家充安装潜力预测 ----
def predict_task2(df: pd.DataFrame):
    """返回 pred (0/1) 和 proba (n×2)。"""
    df, _ = add_task2_features(df, fit=True)
    drop = ["home_charging_available", "ev_adoption_likelihood",
            "target_ord", "task3_anxiety_label"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "task2_catboost.cbm"))
    return model.predict(X), model.predict_proba(X)

# ---- 任务三：里程焦虑识别 ----
def predict_task3(df: pd.DataFrame):
    """返回 pred (0/1) 和 proba (n×2)。"""
    df, _ = add_task3_features(df, fit=True)
    drop = ["task3_anxiety_label", "range_anxiety_score",
            "target_ord", "ev_adoption_likelihood"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "task3_catboost.cbm"))
    return model.predict(X), model.predict_proba(X)


if __name__ == "__main__":
    df = load_processed_data()

    for name, fn in [("任务一", predict_task1),
                      ("任务二", predict_task2),
                      ("任务三", predict_task3)]:
        pred, proba = fn(df)
        print(f"{name}: 预测完成，共 {len(pred)} 条记录")
        print(f"  预测分布: {pd.Series(pred).value_counts().to_dict()}")
