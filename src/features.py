# AI辅助说明：DeepSeek-V4-Pro，v4，DeepSeek，2026-07-29，
# 生成/辅助内容：任务一 8 个业务特征工程函数。
# 该内容经作者核对修改后用于 B 题任务一特征工程。

import numpy as np
import pandas as pd


def add_task1_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    为任务一构造具有业务含义的交互特征。

    Parameters
    ----------
    df : pd.DataFrame
        输入特征 DataFrame（训练集或验证集）。
    fit : bool
        仅用于标记是否基于训练集拟合；当前特征无需训练集统计量，
        保留此参数以保持接口统一。

    Returns
    -------
    pd.DataFrame
        加入新特征后的 DataFrame。
    """
    df = df.copy()
    eps = 1e-6

    # 1. 燃油负担率：当前燃油成本对收入的压力
    df["fuel_burden_ratio"] = (
        12.0 * df["fuel_expense_per_month"] / (df["annual_income"] + eps)
    )

    # 2. 出行一致性：区分规律通勤和额外长途出行
    df["travel_consistency"] = df["weekly_travel_distance_km"] / (
        7.0 * df["daily_commute_km"] + eps
    )

    # 3. 充电便利综合量：同时反映主观便利度和客观距离
    df["charging_convenience"] = df["charging_station_accessibility"] / (
        df["nearest_charging_station_km"] + 1.0
    )

    # 4. 绿色技术倾向：环保与技术接受度的协同作用
    df["green_tech_score"] = (
        df["environmental_awareness_score"] * df["technology_affinity_score"]
    )

    # 5. 认知焦虑差：认知是否能抵消续航焦虑
    df["knowledge_anxiety_gap"] = (
        df["ev_knowledge_score"] - df["range_anxiety_score"]
    )

    # 6. 电动使用成本率：充电费用的经济压力
    df["ev_cost_ratio"] = (
        12.0 * df["monthly_charging_cost"] / (df["annual_income"] + eps)
    )

    # 7. 换车倾向近似：老旧高耗车辆用户可能更愿意换车
    df["replace_tendency"] = (
        df["vehicle_age_years"] * df["fuel_expense_per_month"]
    )

    # 8. 月可支配收入近似：衡量购买能力（与对数收入互补）
    df["monthly_disposable_income"] = df["annual_income"] / 12.0

    # 处理可能出现的 inf（理论上因 eps 已避免，但做兜底）
    df = df.replace([np.inf, -np.inf], np.nan)
    # 对新生成的比值特征，若出现 NaN 用 0 填充（业务上表示无相关压力/倾向）
    new_cols = [
        "fuel_burden_ratio",
        "travel_consistency",
        "charging_convenience",
        "green_tech_score",
        "knowledge_anxiety_gap",
        "ev_cost_ratio",
        "replace_tendency",
        "monthly_disposable_income",
    ]
    for col in new_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return df


def add_task2_features(
    df: pd.DataFrame, fit: bool = True, stats: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    为任务二构造具有业务含义的交互特征。

    Parameters
    ----------
    df : pd.DataFrame
        输入特征 DataFrame（训练集或验证集）。
    fit : bool
        若为 True，基于训练集计算分位数/阈值并保存到 stats；
        若为 False，使用已保存的 stats 构造特征。
    stats : dict | None
        训练集统计量字典，fit=True 时输出，fit=False 时输入。

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (加入新特征后的 DataFrame, stats 字典)
    """
    df = df.copy()
    eps = 1e-6
    if stats is None:
        stats = {}

    # 1. 公共充电稀缺度：公共充电越不方便，家充需求可能越高
    df["charging_scarcity"] = df["nearest_charging_station_km"] / (
        df["charging_station_accessibility"] + 1.0
    )

    # 2. 长通勤标志：日通勤距离是否超过训练集 75% 分位数
    if fit:
        stats["commute_q75"] = float(df["daily_commute_km"].quantile(0.75))
    commute_q75 = stats["commute_q75"]
    df["long_commute_flag"] = (df["daily_commute_km"] > commute_q75).astype(int)

    # 3. 收入分位组：在训练折内对收入四分位分箱（0/1/2/3）
    if fit:
        qs = df["annual_income"].quantile([0.25, 0.5, 0.75]).values
        stats["income_quantiles"] = [float(q) for q in qs]
    q1, q2, q3 = stats["income_quantiles"]
    income_group = pd.cut(
        df["annual_income"],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=False,
    )
    df["income_quantile_group"] = income_group.fillna(0).astype(int)

    # 4. 车辆更新需求：车辆年龄和燃油支出的交互
    df["vehicle_renewal_need"] = (
        df["vehicle_age_years"] * df["fuel_expense_per_month"]
    )

    # 5. 月行驶强度近似：估计月度出行需求
    df["monthly_travel_intensity"] = df["weekly_travel_distance_km"] * 4.33

    # 6. 住宅环境交互：城市类型 × 充电距离、城市类型 × 收入
    df["city_charging_interaction"] = (
        df["city_type"] * df["nearest_charging_station_km"]
    )
    df["city_income_interaction"] = df["city_type"] * df["annual_income"]

    # 兜底：inf -> NaN -> 0
    df = df.replace([np.inf, -np.inf], np.nan)
    new_cols = [
        "charging_scarcity",
        "long_commute_flag",
        "income_quantile_group",
        "vehicle_renewal_need",
        "monthly_travel_intensity",
        "city_charging_interaction",
        "city_income_interaction",
    ]
    for col in new_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return df, stats


def add_task3_features(
    df: pd.DataFrame, fit: bool = True, stats: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    为任务三构造具有业务含义的交互特征。

    Parameters
    ----------
    df : pd.DataFrame
        输入特征 DataFrame（训练集或验证集）。
    fit : bool
        若为 True，基于训练集计算分位数/阈值并保存到 stats；
        若为 False，使用已保存的 stats 构造特征。
    stats : dict | None
        训练集统计量字典，fit=True 时输出，fit=False 时输入。

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (加入新特征后的 DataFrame, stats 字典)
    """
    df = df.copy()
    eps = 1e-6
    if stats is None:
        stats = {}

    # 1. 额外出行强度：近似周末或非通勤长途需求
    df["extra_travel_intensity"] = (
        df["weekly_travel_distance_km"] - 5.0 * df["daily_commute_km"]
    )

    # 2. 日均出行近似：与日通勤距离相互校验
    df["daily_travel_approx"] = df["weekly_travel_distance_km"] / 7.0

    # 3. 充电困难度：同时考虑距离与主观便利度
    df["charging_difficulty"] = df["nearest_charging_station_km"] / (
        df["charging_station_accessibility"] + 1.0
    )

    # 4. 知识缓冲量：知识能否抵消技术与成本担忧
    df["knowledge_buffer"] = (
        df["ev_knowledge_score"] - df["battery_replacement_concern"]
    )

    # 5. 高里程低便利标志：高出行距离且充电便利度低
    if fit:
        stats["weekly_travel_q75"] = float(df["weekly_travel_distance_km"].quantile(0.75))
        stats["charging_access_q25"] = float(
            df["charging_station_accessibility"].quantile(0.25)
        )
    high_travel = df["weekly_travel_distance_km"] > stats["weekly_travel_q75"]
    low_convenience = df["charging_station_accessibility"] < stats["charging_access_q25"]
    df["high_mileage_low_convenience"] = (high_travel & low_convenience).astype(int)

    # 6. 家充缓冲交互：家充是否缓解高里程用户焦虑
    df["home_charging_buffer"] = (
        df["home_charging_available"] * df["daily_commute_km"]
    )

    # 7. 长途占比：反映不规律长途出行
    df["long_distance_ratio"] = df["weekly_travel_distance_km"] / (
        7.0 * df["daily_commute_km"] + eps
    )

    # 兜底：inf -> NaN -> 0
    df = df.replace([np.inf, -np.inf], np.nan)
    new_cols = [
        "extra_travel_intensity",
        "daily_travel_approx",
        "charging_difficulty",
        "knowledge_buffer",
        "high_mileage_low_convenience",
        "home_charging_buffer",
        "long_distance_ratio",
    ]
    for col in new_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return df, stats
