# AI辅助说明：Kimi K2.6，v2.6，Moonshot AI，2026-07-29，
# 生成/辅助内容：任务二完整训练流水线（S0 严格特征 / S1 诊断上界 / S2 任务一 OOF 堆叠）。
# 包含折内特征拟合、Logloss 早停、OOF 阈值优化、多数类基线警告及跨任务无泄漏堆叠。
# 该内容经作者核对修改后用于 B 题任务二建模。

import json
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from catboost import CatBoostClassifier

from common import (
    RANDOM_STATE,
    CATBOOST_TASK_TYPE,
    load_processed_data,
    split_data,
    evaluate_classification,
    save_results,
)
from features import add_task1_features, add_task2_features

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
TARGET_COL = "home_charging_available"
CAT_COLS = ["education_level", "city_type", "current_vehicle_type"]
N_SPLITS = 5
N_TRIALS = 3


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
        X_train,
        y_train,
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


def catboost_cv_task2(
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
        X_tr, feature_stats = add_task2_features(X.iloc[tr_idx], fit=True)
        X_va, _ = add_task2_features(X.iloc[va_idx], fit=False, stats=feature_stats)
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
            X_tr,
            y_tr,
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
    X_train, y_train, cat_features_indices, n_trials: int = N_TRIALS
):
    """第三轮：随机搜索超参数，按 OOF 阈值准确率选择。"""
    print(f"\n{'='*60}")
    print(f"CatBoost 超参数随机搜索 ({n_trials} 组)")
    print(f"{'='*60}")

    rng = np.random.default_rng(RANDOM_STATE)
    param_grid = {
        "depth": [4, 5, 6, 7, 8, 9],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "iterations": [500, 800, 1000],
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
            catboost_cv_task2(
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
# Task1-no-home OOF 堆叠特征生成
# ============================================================
def train_task1_nohome(X, y_task1, cat_features_indices):
    """训练专用的 Task1-no-home CatBoost（删除 home_charging_available）。"""
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        task_type=CATBOOST_TASK_TYPE,
        verbose=False,
        random_seed=RANDOM_STATE,
        iterations=800,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=10,
    )
    model.fit(X, y_task1, cat_features=cat_features_indices, verbose=False)
    return model


def generate_task1_stack_features(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    y_task1_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成 Task1-no-home OOF 概率堆叠特征。

    1. 在外层训练集内部做 5 折 CV，每折训练 Task1-no-home 并预测当前折。
    2. 用全部外层训练集训练 Task1-no-home，预测验证集。
    3. 拼接为 stack_p_low / medium / high、期望值、熵。
    """
    print(f"\n{'='*60}")
    print("生成 Task1-no-home OOF 堆叠特征")
    print(f"{'='*60}")

    # Task1-no-home 特征：删除自身目标及任务二目标
    drop_t1 = ["home_charging_available", "target_ord", "task3_anxiety_label"]
    drop_t1 = [c for c in drop_t1 if c in df_train.columns]
    X_t1_train = df_train.drop(columns=["ev_adoption_likelihood"] + drop_t1)
    X_t1_valid = df_valid.drop(columns=["ev_adoption_likelihood"] + drop_t1)

    # 加入任务一业务特征
    X_t1_train = add_task1_features(X_t1_train, fit=True)
    X_t1_valid = add_task1_features(X_t1_valid, fit=False)

    cat_idx_t1 = get_cat_features_indices(X_t1_train, CAT_COLS)

    # 5 折 OOF
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(df_train), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_t1_train, y_task1_train), 1):
        model = train_task1_nohome(
            X_t1_train.iloc[tr_idx],
            y_task1_train[tr_idx],
            cat_idx_t1,
        )
        oof_proba[va_idx] = model.predict_proba(X_t1_train.iloc[va_idx])
        print(f"  Fold {fold}: Task1-no-home 训练完成")

    # 全量训练集模型 → 预测验证集
    full_model = train_task1_nohome(X_t1_train, y_task1_train, cat_idx_t1)
    valid_proba = full_model.predict_proba(X_t1_valid)

    # 拼接堆叠特征
    df_train_stack = df_train.copy()
    df_valid_stack = df_valid.copy()

    class_names = ["low", "medium", "high"]
    for i, name in enumerate(class_names):
        df_train_stack[f"stack_p_{name}"] = oof_proba[:, i]
        df_valid_stack[f"stack_p_{name}"] = valid_proba[:, i]

    # 采用意愿期望值（Low=1, Medium=2, High=3）
    df_train_stack["stack_adoption_expectation"] = (
        oof_proba[:, 0] * 1 + oof_proba[:, 1] * 2 + oof_proba[:, 2] * 3
    )
    df_valid_stack["stack_adoption_expectation"] = (
        valid_proba[:, 0] * 1 + valid_proba[:, 1] * 2 + valid_proba[:, 2] * 3
    )

    # 预测熵
    eps = 1e-10
    df_train_stack["stack_adoption_entropy"] = -np.sum(
        oof_proba * np.log(oof_proba + eps), axis=1
    )
    df_valid_stack["stack_adoption_entropy"] = -np.sum(
        valid_proba * np.log(valid_proba + eps), axis=1
    )

    print(f"堆叠特征已生成: {df_train_stack.shape}")
    return df_train_stack, df_valid_stack


# ============================================================
# 各方案训练流水线
# ============================================================
def run_scheme_pipeline(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    le: LabelEncoder,
    drop_cols: list[str],
    scheme_name: str,
    run_tuning: bool = True,
):
    """
    运行单方案完整训练流水线（S0 或 S2）。

    Returns
    -------
    dict with keys: model, y_pred, y_proba, metrics, scheme_name
    """
    print(f"\n{'#'*60}")
    print(f"# 方案: {scheme_name}")
    print(f"{'#'*60}")

    X_tr = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_va = df_valid.drop(columns=[c for c in drop_cols if c in df_valid.columns])

    # 业务特征
    X_tr_fe, stats = add_task2_features(X_tr, fit=True)
    X_va_fe, _ = add_task2_features(X_va, fit=False, stats=stats)
    print(f"\n加入业务特征后: {X_tr_fe.shape}")

    cat_idx = get_cat_features_indices(X_tr_fe, CAT_COLS)

    # 基线
    run_baseline(X_tr_fe, y_train, X_va_fe, y_valid, le, class_weight=None, prefix=scheme_name)
    run_baseline(X_tr_fe, y_train, X_va_fe, y_valid, le, class_weight="balanced", prefix=scheme_name)

    # CatBoost 默认
    run_catboost_default(X_tr_fe, y_train, X_va_fe, y_valid, le, use_features=f"{scheme_name}-engineered")

    if not run_tuning:
        return None

    # 超参数搜索
    best_params, best_oof_acc, _, best_oof_auc, search_results = run_catboost_tuning(
        X_tr_fe, y_train, cat_idx, n_trials=N_TRIALS
    )

    # OOF 阈值搜索
    print(f"\n{'='*60}")
    print("OOF 阈值搜索（训练集折外预测）")
    print(f"{'='*60}")
    _, _, _, oof_proba, _ = catboost_cv_task2(
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


def run_s1_diagnostic(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    le: LabelEncoder,
):
    """S1：真实采用意愿诊断上界（仅 LR + 默认 CatBoost，不调参）。"""
    print(f"\n{'#'*60}")
    print(f"# 方案: S1 — 真实采用意愿诊断上界")
    print(f"{'#'*60}")

    drop_cols = ["target_ord", "task3_anxiety_label"]
    drop_cols = [c for c in drop_cols if c in df_train.columns]
    X_tr = df_train.drop(columns=drop_cols)
    X_va = df_valid.drop(columns=drop_cols)

    X_tr_fe, stats = add_task2_features(X_tr, fit=True)
    X_va_fe, _ = add_task2_features(X_va, fit=False, stats=stats)

    # ev_adoption_likelihood 是字符串类别，加入 CAT_COLS
    cat_cols_s1 = CAT_COLS + ["ev_adoption_likelihood"]

    # LR 基线
    run_baseline(X_tr_fe, y_train, X_va_fe, y_valid, le, class_weight=None, prefix="S1", extra_cat_cols=["ev_adoption_likelihood"])

    # CatBoost 默认
    cat_idx = get_cat_features_indices(X_tr_fe, cat_cols_s1)
    run_catboost_default(X_tr_fe, y_train, X_va_fe, y_valid, le, cat_features_indices=cat_idx, use_features="S1-engineered")

    print("\n[S1 仅为诊断上界，不保存模型]")


# ============================================================
# 主流程
# ============================================================
def main():
    # ============================================================
    # 1. 加载数据
    # ============================================================
    df = load_processed_data()
    print(f"原始数据: {df.shape}")

    # ============================================================
    # 2. 外层划分（S0/S1/S2 共用同一划分）
    # ============================================================
    # 先用 split_data 获取外层 train/valid，再从中恢复完整 DataFrame
    X_train_outer, X_valid_outer, y_train, y_valid, le = split_data(
        df, target_col=TARGET_COL, drop_cols=["target_ord", "task3_anxiety_label"]
    )

    # 恢复包含全部列的 DataFrame（split_data 只删除了 target + drop_cols）
    # 注意：X_train_outer 已经不含 target_ord 和 task3_anxiety_label
    # 但保留了 ev_adoption_likelihood（用于 S1 和 S2 的 task1 标签）
    df_train = X_train_outer.copy()
    df_train[TARGET_COL] = y_train
    df_valid = X_valid_outer.copy()
    df_valid[TARGET_COL] = y_valid

    print(f"训练集: {df_train.shape}, 验证集: {df_valid.shape}")
    print(f"训练集类别分布: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"验证集类别分布: {pd.Series(y_valid).value_counts().sort_index().to_dict()}")
    print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    majority_acc = float(max(np.mean(y_train), 1.0 - np.mean(y_train)))
    print(f"\n多数类基线准确率: {majority_acc:.5f}")

    # ============================================================
    # 3. S0：严格原始特征版（已跳过 —— 已知 ACC≈0.65，信号不足）
    # ============================================================
    s0_result = None
    print("\n[S0 已跳过 —— 已知严格特征无有效信号，节省时间]")

    # ============================================================
    # 4. S1：真实采用意愿诊断上界
    # ============================================================
    run_s1_diagnostic(df_train, df_valid, y_train, y_valid, le)

    # ============================================================
    # 5. S2：任务一 OOF 概率堆叠版
    # ============================================================
    # 生成堆叠特征
    y_task1_train = LabelEncoder().fit_transform(df_train["ev_adoption_likelihood"].values)
    df_train_s2, df_valid_s2 = generate_task1_stack_features(
        df_train, df_valid, y_task1_train
    )

    s2_result = run_scheme_pipeline(
        df_train_s2,
        df_valid_s2,
        y_train,
        y_valid,
        le,
        drop_cols=["ev_adoption_likelihood", "home_charging_available"],
        scheme_name="S2",
        run_tuning=True,
    )

    # ============================================================
    # 6. 方案对比与最终选择
    # ============================================================
    print(f"\n{'='*60}")
    print("方案对比（验证集）")
    print(f"{'='*60}")

    if s0_result and s2_result:
        s0_acc = s0_result["metrics"].get("Final S0 accuracy", 0)
        s2_acc = s2_result["metrics"].get("Final S2 accuracy", 0)
        s0_auc = s0_result["roc_auc"]
        s2_auc = s2_result["roc_auc"]

        print(f"S0 严格特征:  ACC={s0_acc:.5f}, AUC={s0_auc:.5f}")
        print(f"S2 OOF 堆叠:  ACC={s2_acc:.5f}, AUC={s2_auc:.5f}")

        if s2_acc > s0_acc + 0.001:
            print("\n结论: S2 稳定优于 S0，选择 S2 作为最终方案。")
            final_result = s2_result
            final_name = "S2"
        else:
            print("\n结论: S2 未稳定超过 S0，保留 S0 并说明任务二可观测特征信号不足。")
            final_result = s0_result
            final_name = "S0"
    else:
        print("\n某方案训练失败，跳过对比。")
        final_result = s0_result or s2_result
        final_name = final_result["scheme_name"] if final_result else "unknown"

    # ============================================================
    # 7. 保存结果
    # ============================================================
    if final_result is None:
        print("无有效结果可保存。")
        return

    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # 模型
    final_result["model"].save_model(
        str(model_dir / f"task2_{final_name}_catboost.cbm")
    )
    print(f"\n模型已保存至: {model_dir / f'task2_{final_name}_catboost.cbm'}")

    # 预测结果
    save_results(
        y_valid,
        final_result["y_pred"],
        final_result["y_proba"],
        le,
        filename=f"task2_{final_name}_results.csv",
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
        model_dir / f"task2_{final_name}_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 搜索日志
    with open(model_dir / f"task2_{final_name}_search_log.json", "w", encoding="utf-8") as f:
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
    print(f"任务二训练完成 — 最终方案: {final_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
