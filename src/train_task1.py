# AI辅助说明：DeepSeek-V4-Pro，v4，DeepSeek，2026-07-29，
# 生成/辅助内容：任务一完整训练流水线，包含数据划分、多项逻辑回归基线、
# CatBoost 主模型、5 折交叉验证、超参数搜索框架及结果保存。
# 该内容经作者核对修改后用于 B 题任务一建模。

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostClassifier

from common import (
    RANDOM_STATE,
    CATBOOST_TASK_TYPE,
    load_processed_data,
    split_data,
    evaluate_classification,
    save_results,
)
from features import add_task1_features

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
TARGET_COL = "ev_adoption_likelihood"
DROP_COLS = ["target_ord"]  # 由目标衍生的列，必须删除
CAT_COLS = ["education_level", "city_type", "current_vehicle_type"]
N_SPLITS = 5
N_TRIALS = 8  # 第一轮快速搜索 trials 数，后续可扩展


def run_baseline(X_train, y_train, X_valid, y_valid, le, class_weight=None):
    """训练并评估多项逻辑回归基线。"""
    print(f"\n{'='*60}")
    print(f"基线模型: 多项逻辑回归 (class_weight={class_weight})")
    print(f"{'='*60}")

    numeric_cols = [c for c in X_train.columns if c not in CAT_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
        ],
        remainder="passthrough",
    )
    pipe = Pipeline([("preprocess", preprocessor), ("clf", LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, n_jobs=5, class_weight=class_weight
    ))])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    t1 = time.time()

    y_pred = pipe.predict(X_valid)
    metrics = evaluate_classification(y_valid, y_pred, le, prefix=f"LR(cw={class_weight}) ")
    print(f"训练耗时: {t1 - t0:.2f}s")
    return pipe, metrics


def run_catboost_default(X_train, y_train, X_valid, y_valid, le, use_features="raw"):
    """第一轮：默认参数 CatBoost，验证管线正确性。"""
    print(f"\n{'='*60}")
    print(f"CatBoost 默认参数 ({use_features})")
    print(f"{'='*60}")

    cat_features_indices = [X_train.columns.get_loc(c) for c in CAT_COLS]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        task_type=CATBOOST_TASK_TYPE,
        verbose=100,
        random_seed=RANDOM_STATE,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        cat_features=cat_features_indices,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=200,
    )
    t1 = time.time()

    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)
    metrics = evaluate_classification(y_valid, y_pred, le, prefix=f"CatBoost(default,{use_features}) ")
    print(f"训练耗时: {t1 - t0:.2f}s")
    return model, metrics, y_proba


def catboost_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    cat_features_indices: list[int],
    n_splits: int = N_SPLITS,
) -> tuple[float, float, list[float]]:
    """
    CatBoost 的 5 折分层交叉验证。

    Returns
    -------
    mean_acc, std_acc, fold_accs
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            task_type=CATBOOST_TASK_TYPE,
            verbose=False,
            random_seed=RANDOM_STATE,
        )
        model.fit(
            X_tr, y_tr,
            cat_features=cat_features_indices,
            eval_set=(X_va, y_va),
            early_stopping_rounds=200,
        )
        y_pred = model.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        fold_accs.append(acc)
        print(f"  Fold {fold}: ACC = {acc:.5f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    return mean_acc, std_acc, fold_accs


def run_catboost_tuning(X_train, y_train, cat_features_indices, n_trials: int = N_TRIALS):
    """第三轮：随机搜索超参数。"""
    print(f"\n{'='*60}")
    print(f"CatBoost 超参数随机搜索 ({n_trials} 组)")
    print(f"{'='*60}")

    rng = np.random.default_rng(RANDOM_STATE)
    param_grid = {
        "depth": [5, 6, 7, 8, 9],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "iterations": [500, 800, 1000, 1500, 2000],
        "l2_leaf_reg": [3, 5, 7, 10, 15],
    }

    best_score = -1.0
    best_params = None
    best_std = 1.0
    results = []

    for i in range(n_trials):
        params = {
            "depth": int(rng.choice(param_grid["depth"])),
            "learning_rate": float(rng.choice(param_grid["learning_rate"])),
            "iterations": int(rng.choice(param_grid["iterations"])),
            "l2_leaf_reg": float(rng.choice(param_grid["l2_leaf_reg"])),
        }
        print(f"\nTrial {i+1}/{n_trials}: {params}")
        mean_acc, std_acc, fold_accs = catboost_cv(
            X_train, y_train, params, cat_features_indices, n_splits=N_SPLITS
        )
        print(f"  5-Fold CV: {mean_acc:.5f} (+/- {std_acc:.5f})")
        results.append({"params": params, "mean_acc": mean_acc, "std_acc": std_acc, "fold_accs": fold_accs})

        # 选择标准：平均准确率最高；若接近则选 std 更小、深度更浅的
        if mean_acc > best_score or (abs(mean_acc - best_score) < 0.0005 and std_acc < best_std):
            best_score = mean_acc
            best_std = std_acc
            best_params = params.copy()

    print(f"\n{'='*60}")
    print(f"最佳 CV 参数: {best_params}")
    print(f"最佳 CV 准确率: {best_score:.5f} (+/- {best_std:.5f})")
    print(f"{'='*60}")
    return best_params, best_score, best_std, results


def main():
    # ============================================================
    # 1. 加载数据
    # ============================================================
    df = load_processed_data()
    print(f"原始数据: {df.shape}")

    # ============================================================
    # 2. 划分数据（删除目标列和下游衍生列）
    # ============================================================
    X_train, X_valid, y_train, y_valid, le = split_data(
        df, target_col=TARGET_COL, drop_cols=DROP_COLS
    )
    print(f"训练集: {X_train.shape}, 验证集: {X_valid.shape}")
    print(f"训练集类别分布: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"验证集类别分布: {pd.Series(y_valid).value_counts().sort_index().to_dict()}")
    print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ============================================================
    # 3. 基线模型
    # ============================================================
    lr_default, _ = run_baseline(X_train, y_train, X_valid, y_valid, le, class_weight=None)
    lr_balanced, _ = run_baseline(X_train, y_train, X_valid, y_valid, le, class_weight="balanced")

    # ============================================================
    # 4. CatBoost 默认参数（原始特征）
    # ============================================================
    cat_features_indices = [X_train.columns.get_loc(c) for c in CAT_COLS]
    cb_default_raw, _, _ = run_catboost_default(X_train, y_train, X_valid, y_valid, le, use_features="raw")

    # ============================================================
    # 5. 特征工程（构造业务特征）
    # ============================================================
    X_train_fe = add_task1_features(X_train, fit=True)
    X_valid_fe = add_task1_features(X_valid, fit=False)
    print(f"\n加入业务特征后: {X_train_fe.shape}")

    cat_features_indices_fe = [X_train_fe.columns.get_loc(c) for c in CAT_COLS]
    cb_default_fe, _, proba_fe = run_catboost_default(
        X_train_fe, y_train, X_valid_fe, y_valid, le, use_features="engineered"
    )

    # ============================================================
    # 6. 类别权重消融（业务特征上）
    # ============================================================
    print(f"\n{'='*60}")
    print("类别权重消融: 平衡权重 vs 平方根平衡权重")
    print(f"{'='*60}")

    for cw_mode in ["balanced", "sqrt_balanced"]:
        classes = np.unique(y_train)
        raw_weights = {c: len(y_train) / (len(classes) * np.sum(y_train == c)) for c in classes}
        if cw_mode == "balanced":
            weights = raw_weights
        else:  # sqrt_balanced
            weights = {c: np.sqrt(w) for c, w in raw_weights.items()}

        cw_list = [weights[c] for c in sorted(weights)]
        mean_acc, std_acc, _ = catboost_cv(
            X_train_fe, y_train,
            {"class_weights": cw_list},
            cat_features_indices_fe,
            n_splits=N_SPLITS,
        )
        print(f"class_weight={cw_mode}: CV_ACC={mean_acc:.5f} (+/- {std_acc:.5f})")

    # ============================================================
    # 7. 超参数搜索（业务特征上）
    # ============================================================
    best_params, best_cv_acc, best_cv_std, search_results = run_catboost_tuning(
        X_train_fe, y_train, cat_features_indices_fe, n_trials=N_TRIALS
    )

    # ============================================================
    # 8. 最终模型：最佳参数在全训练集上重训，在验证集上仅评估一次
    # ============================================================
    print(f"\n{'='*60}")
    print("最终模型：最佳参数 + 全部训练数据")
    print(f"{'='*60}")

    final_model = CatBoostClassifier(
        **best_params,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        task_type=CATBOOST_TASK_TYPE,
        verbose=100,
        random_seed=RANDOM_STATE,
    )
    final_model.fit(
        X_train_fe, y_train,
        cat_features=cat_features_indices_fe,
        verbose=100,
    )

    y_pred_final = final_model.predict(X_valid_fe)
    y_proba_final = final_model.predict_proba(X_valid_fe)
    final_metrics = evaluate_classification(y_valid, y_pred_final, le, prefix="Final ")

    # ============================================================
    # 9. 保存结果
    # ============================================================
    # 模型
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    final_model.save_model(str(model_dir / "task1_catboost.cbm"))
    print(f"模型已保存至: {model_dir / 'task1_catboost.cbm'}")

    # 预测结果
    save_results(y_valid, y_pred_final, y_proba_final, le, filename="task1_results.csv")

    # 特征重要性
    feature_importance = pd.DataFrame({
        "feature": X_train_fe.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print("\nTop 15 特征重要性:")
    print(feature_importance.head(15).to_string(index=False))
    feature_importance.to_csv(model_dir / "task1_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 搜索过程记录
    with open(model_dir / "task1_search_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "best_cv_acc": best_cv_acc,
            "best_cv_std": best_cv_std,
            "search_results": [
                {k: v for k, v in r.items() if k != "fold_accs"} for r in search_results
            ],
        }, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 10. 混淆矩阵深度分析
    # ============================================================
    print(f"\n{'='*60}")
    print("误判分析")
    print(f"{'='*60}")
    cm = confusion_matrix(y_valid, y_pred_final)
    total_per_class = cm.sum(axis=1)
    for i, cls in enumerate(le.classes_):
        wrong = total_per_class[i] - cm[i, i]
        print(f"{cls}: 总 {total_per_class[i]}, 错分 {wrong}, 误分率 {wrong/total_per_class[i]:.3f}")
        for j, cls2 in enumerate(le.classes_):
            if i != j:
                print(f"  -> {cls2}: {cm[i,j]} ({cm[i,j]/total_per_class[i]:.3f})")

    print(f"\n{'='*60}")
    print("任务一训练完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
