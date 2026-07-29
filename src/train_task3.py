# AI辅助说明：Kimi K2.6，v2.6，Moonshot AI，2026-07-29，
# 生成/辅助内容：任务三完整训练流水线（里程焦虑敏感度识别）。
# 包含标签防泄漏处理、主方案与扩展方案消融、OOF 阈值优化、边界样本分析。
# 该内容经作者核对修改后用于 B 题任务三建模。

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
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
from features import add_task3_features

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
TARGET_COL = "task3_anxiety_label"
DROP_COLS = ["range_anxiety_score", "target_ord"]  # 泄漏列和下游衍生列
CAT_COLS = ["education_level", "city_type", "current_vehicle_type"]
N_SPLITS = 5
N_TRIALS_MAIN = 8
N_TRIALS_ABL = 3


# ============================================================
# 公共辅助函数
# ============================================================
def get_cat_features_indices(X: pd.DataFrame, cat_cols: list[str]) -> list[int]:
    """获取分类特征在当前 DataFrame 中的列索引。"""
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]


def run_baseline(
    X_train, y_train, X_valid, y_valid, le,
    class_weight=None, prefix="", extra_cat_cols=None
):
    """训练并评估二元逻辑回归基线。"""
    print(f"\n{'='*60}")
    print(f"基线模型: 二元逻辑回归 (class_weight={class_weight}) [{prefix}]")
    print(f"{'='*60}")

    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    if extra_cat_cols:
        cat_cols.extend([c for c in extra_cat_cols if c in X_train.columns])
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="passthrough",
    )
    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    n_jobs=5,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    t0 = time.time()
    pipe.fit(X_train, y_train)
    t1 = time.time()

    y_pred = pipe.predict(X_valid)
    metrics = evaluate_classification(
        y_valid, y_pred, le, prefix=f"LR(cw={class_weight}) {prefix}"
    )
    print(f"训练耗时: {t1 - t0:.2f}s")
    return pipe, metrics


def run_catboost_default(
    X_train, y_train, X_valid, y_valid, le,
    cat_features_indices=None, use_features="raw", threshold=0.5
):
    """第一轮：默认参数 CatBoost，验证管线正确性。"""
    print(f"\n{'='*60}")
    print(f"CatBoost 默认参数 ({use_features})")
    print(f"{'='*60}")

    if cat_features_indices is None:
        cat_features_indices = get_cat_features_indices(X_train, CAT_COLS)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type=CATBOOST_TASK_TYPE,
        verbose=100,
        random_seed=RANDOM_STATE,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        cat_features=cat_features_indices,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
    )
    t1 = time.time()

    y_proba = model.predict_proba(X_valid)
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    metrics = evaluate_classification(
        y_valid, y_pred, le, prefix=f"CatBoost(default,{use_features}) "
    )

    roc_auc = roc_auc_score(y_valid, y_proba[:, 1])
    print(f"ROC-AUC: {roc_auc:.5f}")
    print(f"训练耗时: {t1 - t0:.2f}s")
    return model, metrics, y_proba


def catboost_cv_task3(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    cat_features_indices: list[int],
    n_splits: int = N_SPLITS,
    threshold: float = 0.5,
) -> tuple[float, float, list[float], np.ndarray, list[int]]:
    """
    CatBoost 的 5 折分层交叉验证（二分类）。
    每折内部重新拟合业务特征，避免统计量泄漏。

    Returns
    -------
    mean_acc, std_acc, fold_accs, oof_proba, fold_best_iterations
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y))
    fold_accs = []
    fold_best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, feature_stats = add_task3_features(X.iloc[tr_idx], fit=True)
        X_va, _ = add_task3_features(X.iloc[va_idx], fit=False, stats=feature_stats)
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type=CATBOOST_TASK_TYPE,
            verbose=False,
            random_seed=RANDOM_STATE,
        )
        model.fit(
            X_tr, y_tr,
            cat_features=cat_features_indices,
            eval_set=(X_va, y_va),
            early_stopping_rounds=100,
        )

        proba = model.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = proba
        best_iteration = model.get_best_iteration()
        fold_best_iterations.append(
            best_iteration + 1
            if best_iteration >= 0
            else int(params.get("iterations", 1000))
        )
        y_pred = (proba >= threshold).astype(int)
        acc = accuracy_score(y_va, y_pred)
        fold_accs.append(acc)
        print(f"  Fold {fold}: ACC = {acc:.5f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    return mean_acc, std_acc, fold_accs, oof_proba, fold_best_iterations


def find_best_threshold(
    oof_proba: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[float, float]:
    """在训练集折外概率上搜索最优分类阈值。"""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    best_acc = -1.0
    best_th = 0.5
    for th in thresholds:
        pred = (oof_proba >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_th, best_acc


def run_catboost_tuning(
    X_train, y_train, cat_features_indices, n_trials: int = N_TRIALS_MAIN
):
    """第三轮：随机搜索超参数，按 OOF 阈值准确率选择。"""
    print(f"\n{'='*60}")
    print(f"CatBoost 超参数随机搜索 ({n_trials} 组)")
    print(f"{'='*60}")

    rng = np.random.default_rng(RANDOM_STATE)
    param_grid = {
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "iterations": [500, 800, 1000, 1500, 1800],
        "l2_leaf_reg": [3, 5, 7, 10, 15],
        "random_strength": [0, 0.5, 1, 1.5, 2],
    }

    best_score = -1.0
    best_params = None
    best_std = 1.0
    best_oof_auc = -1.0
    results = []

    for i in range(n_trials):
        params = {
            "depth": int(rng.choice(param_grid["depth"])),
            "learning_rate": float(rng.choice(param_grid["learning_rate"])),
            "iterations": int(rng.choice(param_grid["iterations"])),
            "l2_leaf_reg": float(rng.choice(param_grid["l2_leaf_reg"])),
            "random_strength": float(rng.choice(param_grid["random_strength"])),
        }
        print(f"\nTrial {i+1}/{n_trials}: {params}")
        mean_acc, std_acc, fold_accs, oof_proba, fold_best_iterations = (
            catboost_cv_task3(
                X_train, y_train, params, cat_features_indices, n_splits=N_SPLITS
            )
        )
        oof_auc = roc_auc_score(y_train, oof_proba)
        trial_threshold, threshold_acc = find_best_threshold(oof_proba, y_train)
        median_best_iteration = int(np.median(fold_best_iterations))
        print(
            f"  5-Fold CV: ACC={mean_acc:.5f} (+/- {std_acc:.5f}), "
            f"AUC={oof_auc:.5f}, threshold_ACC={threshold_acc:.5f} "
            f"(t={trial_threshold:.2f}), median_iter={median_best_iteration}"
        )
        results.append(
            {
                "params": params,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "oof_auc": oof_auc,
                "best_threshold": trial_threshold,
                "oof_acc_at_best_threshold": threshold_acc,
                "median_best_iteration": median_best_iteration,
                "fold_accs": fold_accs,
                "fold_best_iterations": fold_best_iterations,
            }
        )

        if threshold_acc > best_score or (
            np.isclose(threshold_acc, best_score, atol=1e-12)
            and oof_auc > best_oof_auc
        ):
            best_score = threshold_acc
            best_std = std_acc
            best_oof_auc = oof_auc
            best_params = params.copy()
            best_params["iterations"] = median_best_iteration

    print(f"\n{'='*60}")
    print(f"最佳 CV 参数: {best_params}")
    print(f"最佳 OOF 阈值准确率: {best_score:.5f}")
    print(f"最佳参数 OOF AUC: {best_oof_auc:.5f}")
    majority_acc = float(max(np.mean(y_train), 1.0 - np.mean(y_train)))
    print(f"多数类基线准确率: {majority_acc:.5f}")
    if best_score <= majority_acc + 0.001:
        print("警告: 最佳模型未显著超过多数类基线，当前特征集可能缺乏有效信号。")
    print(f"{'='*60}")
    return best_params, best_score, best_std, best_oof_auc, results


# ============================================================
# 方案训练流水线
# ============================================================
def run_scheme_pipeline(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    le,
    cat_cols: list[str],
    scheme_name: str,
    n_trials: int = N_TRIALS_MAIN,
):
    """
    运行单方案完整训练流水线。

    Returns
    -------
    dict with keys: model, y_pred, y_proba, metrics, scheme_name, ...
    """
    print(f"\n{'#'*60}")
    print(f"# 方案: {scheme_name}")
    print(f"{'#'*60}")

    # 业务特征
    X_tr_fe, stats = add_task3_features(df_train, fit=True)
    X_va_fe, _ = add_task3_features(df_valid, fit=False, stats=stats)
    print(f"\n加入业务特征后: {X_tr_fe.shape}")

    cat_idx = get_cat_features_indices(X_tr_fe, cat_cols)

    # 超出标准 CAT_COLS 的分类特征需显式传给逻辑回归管线
    extra_cat_cols = [c for c in cat_cols if c not in CAT_COLS] or None

    # 基线
    run_baseline(X_tr_fe, y_train, X_va_fe, y_valid, le, class_weight=None, prefix=scheme_name, extra_cat_cols=extra_cat_cols)
    run_baseline(X_tr_fe, y_train, X_va_fe, y_valid, le, class_weight="balanced", prefix=scheme_name, extra_cat_cols=extra_cat_cols)

    # CatBoost 默认
    run_catboost_default(X_tr_fe, y_train, X_va_fe, y_valid, le, cat_features_indices=cat_idx, use_features=f"{scheme_name}-engineered")

    # 超参数搜索
    best_params, best_oof_acc, _, best_oof_auc, search_results = run_catboost_tuning(
        X_tr_fe, y_train, cat_idx, n_trials=n_trials
    )

    # OOF 阈值搜索
    print(f"\n{'='*60}")
    print("OOF 阈值搜索（训练集折外预测）")
    print(f"{'='*60}")
    _, _, _, oof_proba, _ = catboost_cv_task3(
        X_tr_fe, y_train, best_params, cat_idx, n_splits=N_SPLITS
    )
    best_th, oof_acc_th = find_best_threshold(oof_proba, y_train)
    print(f"最优阈值: {best_th:.3f}, OOF ACC: {oof_acc_th:.5f}")

    # 最终模型
    print(f"\n{'='*60}")
    print(f"最终模型：{scheme_name} 最佳参数 + 全部训练数据")
    print(f"{'='*60}")

    final_model = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type=CATBOOST_TASK_TYPE,
        verbose=100,
        random_seed=RANDOM_STATE,
    )
    final_model.fit(X_tr_fe, y_train, cat_features=cat_idx, verbose=100)

    y_proba = final_model.predict_proba(X_va_fe)
    y_pred = (y_proba[:, 1] >= best_th).astype(int)
    final_metrics = evaluate_classification(y_valid, y_pred, le, prefix=f"Final {scheme_name} ")
    roc_auc = roc_auc_score(y_valid, y_proba[:, 1])
    print(f"ROC-AUC: {roc_auc:.5f}")

    return {
        "model": final_model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": final_metrics,
        "roc_auc": roc_auc,
        "best_params": best_params,
        "best_threshold": best_th,
        "oof_acc": oof_acc_th,
        "oof_auc": best_oof_auc,
        "scheme_name": scheme_name,
        "X_train_fe": X_tr_fe,
        "X_valid_fe": X_va_fe,
        "search_results": search_results,
    }


def boundary_analysis(y_true, y_pred, scores):
    """
    对原始评分接近阈值的样本进行误差分析。
    仅用于解释，不能将原始评分重新作为模型输入。
    """
    print(f"\n{'='*60}")
    print("边界样本分析（原始评分 4/5/6/7）")
    print(f"{'='*60}")
    for s in [4, 5, 6, 7]:
        mask = scores == s
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        err_rate = 1.0 - acc
        print(f"  评分={s}: 样本数={mask.sum()}, 准确率={acc:.4f}, 错误率={err_rate:.4f}")


# ============================================================
# 主流程
# ============================================================
def main():
    # ============================================================
    # 1. 加载数据
    # ============================================================
    df = load_processed_data()
    print(f"原始数据: {df.shape}")

    # 保存边界分析所需的原始评分（划分后通过索引对应）
    boundary_scores = df["range_anxiety_score"].copy()

    # 删除泄漏列和下游衍生列
    df_model = df.drop(columns=DROP_COLS)

    # ============================================================
    # 2. 分层划分
    # ============================================================
    X_train, X_valid, y_train, y_valid, le = split_data(
        df_model, target_col=TARGET_COL, drop_cols=[]
    )

    # 通过索引取回验证集对应的原始评分
    valid_scores = boundary_scores.loc[X_valid.index].values

    print(f"训练集: {X_train.shape}, 验证集: {X_valid.shape}")
    print(f"训练集类别分布: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"验证集类别分布: {pd.Series(y_valid).value_counts().sort_index().to_dict()}")
    print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    majority_acc = float(max(np.mean(y_train), 1.0 - np.mean(y_train)))
    print(f"\n多数类基线准确率: {majority_acc:.5f}")

    skip_main = os.environ.get("TASK3_MODE", "") == "ablation"
    skip_abl  = os.environ.get("TASK3_MODE", "") == "main"

    # ============================================================
    # 3. 主方案：不使用 ev_adoption_likelihood
    # ============================================================
    if skip_main:
        print(f"\n{'#'*60}")
        print(f"# TASK3_MODE=ablation, 跳过 Main 方案")
        print(f"{'#'*60}")
        main_result = None
    else:
        X_train_main = X_train.drop(columns=["ev_adoption_likelihood"])
        X_valid_main = X_valid.drop(columns=["ev_adoption_likelihood"])

        main_result = run_scheme_pipeline(
            X_train_main, X_valid_main, y_train, y_valid, le,
            cat_cols=CAT_COLS, scheme_name="Main", n_trials=N_TRIALS_MAIN,
        )

    # ============================================================
    # 4. 消融方案：加入 ev_adoption_likelihood
    # ============================================================
    if skip_abl:
        print(f"\n{'#'*60}")
        print(f"# TASK3_MODE=main, 跳过 Ablation 方案")
        print(f"{'#'*60}")
        abl_result = None
    else:
        cat_cols_abl = CAT_COLS + ["ev_adoption_likelihood"]
        abl_result = run_scheme_pipeline(
            X_train, X_valid, y_train, y_valid, le,
            cat_cols=cat_cols_abl, scheme_name="Ablation", n_trials=N_TRIALS_ABL,
        )

    # ============================================================
    # 5. 方案对比
    # ============================================================
    main_acc = main_result["metrics"].get("Final Main accuracy", 0) if main_result else None
    main_auc = main_result["roc_auc"] if main_result else None
    abl_acc  = abl_result["metrics"].get("Final Ablation accuracy", 0) if abl_result else None
    abl_auc  = abl_result["roc_auc"] if abl_result else None

    if main_result is not None and abl_result is not None:
        print(f"\n{'='*60}")
        print("方案对比（验证集）")
        print(f"{'='*60}")
        print(f"Main     (无采用意愿): ACC={main_acc:.5f}, AUC={main_auc:.5f}")
        print(f"Ablation (有采用意愿): ACC={abl_acc:.5f}, AUC={abl_auc:.5f}")
        if abl_acc > main_acc + 0.001:
            print("\n结论: 消融方案优于主方案，说明 ev_adoption_likelihood 提供了额外信号。")
            print("但主方案更具可部署性（预测时无需任务一结果）。")
        else:
            print("\n结论: 消融方案未稳定超过主方案，ev_adoption_likelihood 并非必要特征。")
    elif main_result is not None:
        print(f"\nMain (无采用意愿): ACC={main_acc:.5f}, AUC={main_auc:.5f}")
    elif abl_result is not None:
        print(f"\nAblation (有采用意愿): ACC={abl_acc:.5f}, AUC={abl_auc:.5f}")

    # 选择主方案作为最终模型（可部署性优先）；若无 Main，退而用 Ablation
    if main_result is not None:
        final_result = main_result
        final_name = "Main"
    else:
        final_result = abl_result
        final_name = "Ablation"

    # ============================================================
    # 6. 边界样本分析
    # ============================================================
    boundary_analysis(y_valid, final_result["y_pred"], valid_scores)

    # ============================================================
    # 7. 保存结果
    # ============================================================
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # 模型
    final_result["model"].save_model(
        str(model_dir / "task3_catboost.cbm")
    )
    print(f"\n模型已保存至: {model_dir / 'task3_catboost.cbm'}")

    # 预测结果
    save_results(
        y_valid,
        final_result["y_pred"],
        final_result["y_proba"],
        le,
        filename="task3_results.csv",
    )

    # 特征重要性
    feature_importance = pd.DataFrame(
        {
            "feature": final_result["X_train_fe"].columns,
            "importance": final_result["model"].get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 15 特征重要性:")
    print(feature_importance.head(15).to_string(index=False))
    feature_importance.to_csv(
        model_dir / "task3_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 搜索日志
    with open(model_dir / "task3_search_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme": final_name,
                "best_params": final_result["best_params"],
                "best_threshold": final_result["best_threshold"],
                "oof_acc": final_result["oof_acc"],
                "oof_auc": final_result["oof_auc"],
                "final_val_accuracy": final_result["metrics"].get(
                    f"Final {final_name} accuracy"
                ),
                "final_roc_auc": final_result["roc_auc"],
                "search_results": [
                    {k: v for k, v in r.items() if k not in ("fold_accs", "fold_best_iterations")}
                    for r in final_result["search_results"]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 混淆矩阵深度分析
    print(f"\n{'='*60}")
    print("误判分析")
    print(f"{'='*60}")
    cm = confusion_matrix(y_valid, final_result["y_pred"])
    total_per_class = cm.sum(axis=1)
    for i, cls in enumerate(le.classes_):
        wrong = total_per_class[i] - cm[i, i]
        print(
            f"{cls}: 总 {total_per_class[i]}, 错分 {wrong}, 误分率 {wrong/total_per_class[i]:.3f}"
        )
        for j, cls2 in enumerate(le.classes_):
            if i != j:
                print(f"  -> {cls2}: {cm[i,j]} ({cm[i,j]/total_per_class[i]:.3f})")

    print(f"\n{'='*60}")
    print(f"任务三训练完成 — 最终方案: {final_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
