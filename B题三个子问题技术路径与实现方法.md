# B题三个子问题技术路径与实现方法

## 1. 问题概述

赛题要求基于 50,000 条电动汽车消费者记录完成三个分类任务：

| 子问题 | 预测目标 | 任务类型 | 评分权重 |
|---|---|---:|---:|
| 任务一：电动汽车采用意愿分类 | `ev_adoption_likelihood` | Low/Medium/High 三分类 | 20% |
| 任务二：家庭充电设施安装潜力预测 | `home_charging_available` | 0/1 二分类 | 40% |
| 任务三：里程焦虑敏感度识别 | `range_anxiety_score > 5` | 0/1 二分类 | 40% |

最终成绩为：

\[
Score=ACC_1\times100\times0.2+ACC_2\times100\times0.4+ACC_3\times100\times0.4
\]

因此，三个任务均以准确率为主要优化指标。任务二和任务三合计占 80%，应优先投入特征分析与调参资源；同时记录宏平均 F1、平衡准确率和混淆矩阵，防止模型仅偏向多数类。

## 2. 总体技术路线

### 2.1 数据核查结论

对原始 CSV 的初步检查结果如下：

- 数据共 50,000 行、23 列，无完全重复行。
- `education_level`、`charging_station_accessibility`、`ev_knowledge_score` 各缺失 500 条。
- 三列缺失记录并不完全重合，共有 1,489 行至少含一个缺失值。
- `fuel_expense_per_month` 中有 271 个负值，不符合燃油支出的业务含义。
- 任务一类别分布为 High 29,670、Medium 12,078、Low 8,252，存在类别不均衡。
- 任务二类别分布为 1 类 32,488、0 类 17,512，存在轻度类别不均衡。
- 按 `range_anxiety_score > 5` 构造任务三标签后，0 类 25,395、1 类 24,605，类别基本均衡。

### 2.2 统一建模流程

三个任务采用相同的实验框架，以保证结果可复现和公平比较：

1. 使用固定随机种子 2026，按目标变量进行分层抽样，划分 80% 训练集和 20% 独立验证集。
2. 仅在 80% 训练集内部执行 5 折分层交叉验证，用于模型选择和超参数优化。
3. 缺失值填补、异常值边界、编码器和特征选择器均只在当前训练折拟合，再作用于验证折，禁止利用验证集统计量。
4. 每项任务先训练可解释的线性基线，再训练 CatBoost 等非线性树模型。
5. 以交叉验证平均准确率选择模型，以宏平均 F1 和混淆矩阵辅助判断少数类效果。
6. 模型确定后，在独立验证集上只评估一次；最终用相同参数在全部可训练数据上重训并保存。

推荐将 CatBoost 作为主模型，原因是它能直接处理分类特征、缺失值和非线性关系，并能减少独热编码带来的维度扩张。LightGBM 或 XGBoost 可作为对照模型；若环境中不增加第三方梯度提升库，则可使用 scikit-learn 的 HistGradientBoosting 或 RandomForest 作为替代。

### 2.3 通用数据预处理

#### 缺失值

- 线性模型：数值变量使用训练折中位数填补，并增加缺失指示变量；分类变量填为 `Missing` 后进行独热编码。
- CatBoost：分类变量缺失填为字符串 `Missing`；数值缺失可交由模型处理，也可使用训练折中位数填补并保留缺失指示变量。
- 对三列缺失是否具有共同机制进行分组统计，比较缺失组和非缺失组的目标比例。若目标比例有明显差异，保留缺失指示变量。

#### 异常值

- 将 `fuel_expense_per_month < 0` 视为无效记录并置为缺失，再按训练折中位数填补，同时构造 `fuel_expense_invalid` 指示变量。
- 对收入、通勤距离、燃油支出等连续变量绘制箱线图和分位数图。线性模型可按训练折的 1% 和 99% 分位数缩尾，树模型原则上保留合理的极端值。
- 不直接删除异常行，以免减少样本量或改变类别分布。

#### 变量编码和缩放

- 分类变量：`education_level`、`city_type`、`current_vehicle_type`。
- 二值变量保持 0/1。
- 线性模型对连续变量进行标准化；树模型无需标准化。
- `education_level` 可以同时尝试普通分类编码和有序编码，但是否采用有序编码必须由交叉验证结果决定。

#### 防止信息泄漏

- 任何任务都必须从输入中删除该任务的目标列。
- 任务三必须先由原始评分生成标签，然后从特征中彻底删除 `range_anxiety_score`，不能使用它构造均值、分箱或交互项。
- 所有预处理必须封装在 sklearn `Pipeline`/`ColumnTransformer` 或 CatBoost 的单一训练流程中。
- 不使用独立验证集标签进行阈值搜索、特征选择或超参数调整。
- 对跨任务目标变量进行严格版和扩展版对照。若某变量在实际预测时不可获得，或属于预测目标的下游结果，即使能提高本地准确率也不应使用。

## 3. 子问题一：电动汽车采用意愿分类

### 3.1 建模目标与难点

目标是预测 `ev_adoption_likelihood` 的 Low、Medium、High 三个类别。主要难点是 High 类占比约 59.3%，Low 类仅约 16.5%；若只优化总体准确率，模型可能牺牲 Low 和 Medium 类。

三类标签虽然具有自然顺序，但首先应按普通三分类处理，避免强制假设各等级之间距离相等。可将有序分类模型作为补充实验，但最终仍以验证准确率选择方案。

### 3.2 推荐特征

基础特征可分为以下几组：

- 人口属性：`age`、`annual_income`、`education_level`、`city_type`。
- 当前用车与成本：`current_vehicle_type`、`vehicle_age_years`、`fuel_expense_per_month`。
- 出行需求：`daily_commute_km`、`weekly_travel_distance_km`。
- 充电条件：`charging_station_accessibility`、`nearest_charging_station_km`、`home_charging_available`、`electricity_cost_per_kwh`。
- 认知与态度：`environmental_awareness_score`、`government_incentive_awareness`、`technology_affinity_score`、`range_anxiety_score`、`battery_replacement_concern`、`ev_knowledge_score`、`previous_ev_experience`。
- 预估使用成本：`monthly_energy_consumption_kwh`、`monthly_charging_cost`。

必须删除目标列 `ev_adoption_likelihood`。

### 3.3 特征工程

建议构造以下具有业务含义的特征：

| 特征 | 计算方法 | 含义 |
|---|---|---|
| 燃油负担率 | `12 * fuel_expense_per_month / annual_income` | 当前燃油成本对收入的压力 |
| 月可支配收入近似值 | `annual_income / 12` | 衡量购买能力 |
| 出行一致性 | `weekly_travel_distance_km / (7 * daily_commute_km + eps)` | 区分规律通勤和额外长途出行 |
| 充电便利综合量 | `charging_station_accessibility / (nearest_charging_station_km + 1)` | 同时反映主观便利度和客观距离 |
| 绿色技术倾向 | 环保得分与技术接受度的乘积或均值 | 反映环保和技术偏好的协同作用 |
| 认知焦虑差 | `ev_knowledge_score - range_anxiety_score` | 认知是否能够抵消续航焦虑 |
| 电动使用成本率 | `12 * monthly_charging_cost / annual_income` | 充电费用的经济压力 |
| 换车倾向近似 | 车辆年龄与燃油支出的交互项 | 老旧高耗车辆用户可能更愿意换车 |

比值计算统一加入很小的 `eps` 防止除零。对树模型无需一次性加入大量高阶交互，只保留有明确业务含义且能通过交叉验证验证的特征。

### 3.4 模型方案

#### 基线模型

采用多项逻辑回归：

- 分类变量独热编码，数值变量中位数填补并标准化。
- 调整正则化强度 `C`。
- 对比不加类别权重、`class_weight='balanced'` 两种设置。

该模型可提供方向明确的系数，适合作为论文中的机制解释和性能下限。

#### 主模型

采用 CatBoostClassifier：

- 损失函数：`MultiClass`。
- 评估指标：`Accuracy`，同时记录 `TotalF1:average=Macro`。
- 初始参数范围：`depth=5~9`、`learning_rate=0.02~0.1`、`iterations=500~2000`、`l2_leaf_reg=3~15`。
- 使用早停，早停轮数可设为 100~200。
- 类别权重分别尝试不加权、平衡权重和平方根平衡权重。由于最终指标是准确率，不能默认加权一定更优。

可训练 LightGBM/XGBoost 作为第二模型。如果两个模型的折外预测错误互补，可按交叉验证结果对预测概率加权融合；权重只能根据训练集折外预测确定。

### 3.5 评估与解释

- 主指标：准确率 `ACC1`。
- 辅助指标：宏平均 F1、每类召回率、三分类混淆矩阵。
- 解释方法：CatBoost 特征重要性、Permutation Importance；最终可对最佳模型使用 SHAP 分析全局重要性及典型用户的个体预测原因。
- 重点检查 Low 是否经常被误判为 High，以及 Medium 是否成为最难识别的过渡类别。

### 3.6 实施步骤

1. 构建三分类标签并分层划分数据。
2. 建立多项逻辑回归基线。
3. 加入业务特征并训练 CatBoost。
4. 在 5 折交叉验证中比较原始特征、业务特征和类别权重。
5. 固定最佳参数，在独立验证集计算 `ACC1` 和混淆矩阵。
6. 保存模型、特征列表、标签顺序和预测 CSV。

## 4. 子问题二：家庭充电设施安装潜力预测

### 4.1 建模目标与难点

目标是预测 `home_charging_available`。1 类占约 65.0%，0 类占约 35.0%。任务的核心是从居住环境、周边充电设施、经济条件和出行需求中识别具备家庭充电条件或意愿的用户。

赛题目标名称更接近“当前是否具备家充条件”，而描述中同时包含“安装意愿”。论文中应明确：模型实际学习的是数据列所表示的当前家充可用状态，并将预测为 1 的用户解释为较高安装潜力人群，避免扩大标签含义。

### 4.2 EDV/EDA：先验证特征来源是否有效

任务二不再直接从“模型调参”开始，而是先执行 EDV/EDA（探索性数据验证与分析），判断当前特征来源是否含有可学习信号。已有诊断结果如下：

| 特征/模型方案 | 验证 ACC | ROC-AUC | 结论 |
|---|---:|---:|---|
| 多数类基线（全部预测为 1） | 0.6498 | 0.5000 | 任务二最低基准 |
| 严格特征 + Logistic Regression | 0.6498 | 0.4955 | 基本无有效排序信号 |
| 严格特征 + ExtraTrees | 0.6485 | 0.4973 | 换模型不能解决问题 |
| 严格特征 + HistGradientBoosting | 0.6498 | 0.4935 | 换模型不能解决问题 |
| 严格特征 + CatBoost | 0.6498 | 0.4947 | 退化为几乎全部预测 1 |
| 加入真实采用意愿的诊断上界 + LR | 0.6668 | 0.6157 | 特征来源改变后出现有效信号 |
| 加入真实采用意愿的诊断上界 + CatBoost | 0.6694 | 0.6160 | CatBoost 仍有约 2 个百分点提升空间 |

严格特征中，除跨任务衍生列外的单变量绝对相关系数大多低于 0.013；而 `target_ord` 与家充标签的相关系数约为 0.095。按真实采用意愿分组后，家充比例分别为：High 68.4%、Medium 62.4%、Low 56.4%。因此，任务二当前瓶颈主要是**特征来源缺乏信号**，而不是 CatBoost 模型选择错误。

上述“真实采用意愿”结果仅用于估计信号上界，不能直接作为最终部署方案，也不能在独立验证集上继续反复调参。最终方案必须通过训练集 OOF 预测构造可部署的跨任务特征。

### 4.3 三类特征来源方案

必须从所有方案中删除任务二目标 `home_charging_available`。在此基础上设计三组对照：

#### S0：严格原始特征版

- 居住条件：`city_type`。
- 公共充电条件：`nearest_charging_station_km`、`charging_station_accessibility`。
- 经济与教育：`annual_income`、`education_level`、`electricity_cost_per_kwh`。
- 当前车辆：`current_vehicle_type`、`vehicle_age_years`、`fuel_expense_per_month`。
- 出行需求：`daily_commute_km`、`weekly_travel_distance_km`。
- 用户认知：`ev_knowledge_score`、`previous_ev_experience`、政府补贴认知和技术接受度。
- 删除 `ev_adoption_likelihood`、`target_ord` 和 `task3_anxiety_label`。

S0 是无跨任务信息的安全基线。现有实验显示其上限接近多数类准确率，但仍需保留作为消融对照。

#### S1：真实采用意愿上界版

在 S0 上加入真实 `ev_adoption_likelihood`，仅用于回答“若采用意愿在预测时已知，任务二最多能提升多少”。S1 不作为默认最终模型；只有题目流程明确保证预测阶段可获得真实采用意愿标签时，才允许进入候选方案。

#### S2：任务一 OOF 概率堆叠版（推荐主方案）

不使用真实采用意愿标签，改由任务一模型生成以下概率特征：

- `stack_p_low`：任务一预测为 Low 的概率。
- `stack_p_medium`：任务一预测为 Medium 的概率。
- `stack_p_high`：任务一预测为 High 的概率。
- `stack_adoption_expectation = 1*p_low + 2*p_medium + 3*p_high`：采用意愿期望值。
- `stack_adoption_entropy = -sum(p_c * log(p_c + eps))`：任务一预测不确定度。

为防止任务二标签通过第一阶段模型泄漏，生成堆叠特征的任务一模型必须删除 `home_charging_available`，即建立专用的“Task1-no-home”模型。具体生成规则为：

1. 在任务二外层训练集内部进行 5 折交叉拟合；每折用另外 4 折训练 Task1-no-home，并预测当前折，拼成完整 OOF 概率。
2. 对任务二独立验证集，只能使用在全部外层训练集上训练的 Task1-no-home 模型进行前向预测，禁止读取验证集真实 `ev_adoption_likelihood` 或 `home_charging_available`。
3. 最终测试阶段先运行 Task1-no-home，再将预测概率输入任务二 CatBoost，形成可复现的两阶段推理链。

### 4.4 特征工程

#### 原始业务特征

| 特征 | 计算方法 | 含义 |
|---|---|---|
| 公共充电稀缺度 | `nearest_charging_station_km / (charging_station_accessibility + 1)` | 公共充电越不方便，家充需求可能越高 |
| 长通勤标志 | 日通勤距离是否超过训练折的 75% 分位数 | 识别高频充电需求用户 |
| 收入分位组 | 在训练折内对收入分箱 | 表示安装支付能力，防止固定阈值失真 |
| 车辆更新需求 | 车辆年龄和燃油支出的交互 | 反映换车和配套安装机会 |
| 住宅环境交互 | `city_type` 与最近充电站距离、收入的交互 | 不同城市类型的基础设施含义不同 |
| 月行驶强度近似 | `weekly_travel_distance_km * 4.33` | 估计月度出行需求 |

收入分箱边界和长通勤阈值必须在每个训练折内计算，再作用于对应验证折。禁止在完成特征工程后再进行交叉验证。

#### 堆叠交互特征

在 S2 中优先保留任务一概率原值，让 CatBoost 自动学习交互；若 OOF 消融显示稳定提升，再尝试：

- `stack_p_high * charging_scarcity`：高采用意愿与公共充电稀缺的协同作用。
- `stack_adoption_expectation * annual_income`：采用意愿与安装支付能力的交互。
- `stack_p_high * previous_ev_experience`：采用意愿与既往 EV 经验的交互。

### 4.5 模型方案

#### 基线模型

采用二元逻辑回归验证各特征来源：

- 分别训练 S0、S1、S2 三版，比较 ACC 和 ROC-AUC。
- 预处理采用中位数填补、独热编码和标准化。
- 比较普通权重与 `class_weight='balanced'`，但最终仍按准确率选择。

#### 主模型

主模型保持 CatBoostClassifier 不变：

- 损失函数：`Logloss`；使用 `Logloss` 进行早停，避免 `Accuracy` 在多数类平台上过早停止。
- 交叉验证中记录固定 0.5 阈值 ACC、OOF ROC-AUC、最优 OOF 阈值 ACC 和最佳迭代数。
- 收入分箱、通勤阈值和堆叠概率必须在当前训练折内部拟合或生成。
- 最终迭代数取最佳参数组各折最佳迭代数的中位数，保证 CV 与最终训练一致。
- 超参数范围保持 `depth=4~9`、`learning_rate=0.02~0.1`、`iterations=500~2000`、`l2_leaf_reg=3~15`、`random_strength=0~2`；在 S2 被证明有效前，不盲目扩大搜索次数。

参数选择优先比较训练集 OOF 最优阈值准确率；若准确率相同，再比较 OOF ROC-AUC、折间波动和模型复杂度。阈值只能由训练集 OOF 概率确定，不能在独立验证集上搜索。

### 4.6 评估与解释

- 主指标：准确率 `ACC2`。
- 辅助指标：F1、平衡准确率、ROC-AUC、混淆矩阵、预测为 0/1 的样本数。
- 必须报告多数类基线；若模型 ACC 未超过基线 0.1 个百分点或 ROC-AUC 仍约为 0.5，应判定当前特征来源无效，而不是继续扩大调参。
- S0、S1、S2 使用完全相同的外层训练/验证划分；S1 只作为信号上界，最终结论重点比较 S0 与 S2。
- 使用 SHAP 分析 `stack_p_*`、收入、充电稀缺度及其交互，说明采用意愿预测如何影响家充判断。
- 业务上同时关注假阳性：将不具备条件的用户误判为潜力用户会增加营销成本。

### 4.7 新实施步骤

1. 固定 80/20 分层外层划分，独立验证集在最终方案确定前保持封存。
2. 运行 S0 严格特征版，确认多数类基线、LR、CatBoost 和 OOF ROC-AUC。
3. 在训练集内部运行 S1 上界诊断，确认跨任务采用意愿是否提供稳定信号；S1 不直接作为可部署结论。
4. 构建删除 `home_charging_available` 的 Task1-no-home 模型，生成训练集 5 折 OOF 概率和验证集前向概率。
5. 将概率特征加入 S0，形成 S2；依次进行 LR 对照、CatBoost 训练和堆叠特征消融。
6. 在 S2 训练集 OOF 概率上确定参数、最佳迭代数和分类阈值。
7. 仅对最终选定的 S0 或 S2 在独立验证集评估一次，计算 `ACC2`、ROC-AUC、混淆矩阵并输出逐行概率。
8. 若 S2 未稳定超过 S0，则保留 S0 并说明任务二可观测特征信号不足；若 S2 稳定提升，则将两阶段推理链与无泄漏证明写入论文和支撑材料。

## 5. 子问题三：里程焦虑敏感度识别

### 5.1 标签构造与难点

按赛题建议构造二分类标签：

\[
y_3=\begin{cases}
1,&range\_anxiety\_score>5\\
0,&range\_anxiety\_score\leq5
\end{cases}
\]

构造后 0 类有 25,395 条，1 类有 24,605 条，类别基本均衡。最大风险不是不均衡，而是标签泄漏：生成标签后必须删除原始 `range_anxiety_score`，也不能保留任何由该评分直接计算的特征。

### 5.2 推荐特征

必须删除 `range_anxiety_score`。优先使用：

- 出行压力：`daily_commute_km`、`weekly_travel_distance_km`。
- 充电条件：`charging_station_accessibility`、`nearest_charging_station_km`、`home_charging_available`。
- 认知和经验：`ev_knowledge_score`、`previous_ev_experience`、`technology_affinity_score`。
- 风险态度：`battery_replacement_concern`。
- 成本因素：`fuel_expense_per_month`、`electricity_cost_per_kwh`、`monthly_charging_cost`。
- 人口和车辆属性：年龄、收入、学历、城市类型、车型和车龄。

主方案不使用 `ev_adoption_likelihood`：采用意愿可能受到里程焦虑直接影响，属于潜在下游变量。将其加入扩展方案做消融实验，如果真实预测流程无法获得该变量，则无论本地分数是否提高都应弃用。

### 5.3 特征工程

| 特征 | 计算方法 | 含义 |
|---|---|---|
| 额外出行强度 | `weekly_travel_distance_km - 5 * daily_commute_km` | 近似周末或非通勤长途需求 |
| 日均出行近似 | `weekly_travel_distance_km / 7` | 与日通勤距离相互校验 |
| 充电困难度 | `nearest_charging_station_km / (charging_station_accessibility + 1)` | 同时考虑距离与主观便利度 |
| 知识缓冲量 | `ev_knowledge_score - battery_replacement_concern` | 知识能否抵消技术与成本担忧 |
| 高里程低便利标志 | 高出行距离且充电便利度低 | 识别焦虑高风险组合 |
| 家充缓冲交互 | `home_charging_available` 与出行距离的交互 | 家充是否缓解高里程用户焦虑 |
| 长途占比 | `weekly_travel_distance_km / (7 * daily_commute_km + eps)` | 反映不规律长途出行 |

“高出行”和“低便利”的阈值应使用训练折分位数确定，不能读取验证折分布。

### 5.4 模型方案

#### 基线模型

采用二元逻辑回归，重点验证各因素的方向：

- 出行距离、充电困难度和电池担忧预计提高高焦虑概率。
- 家充条件、EV 知识和既往 EV 经验可能降低高焦虑概率。
- 实际方向以数据估计和置信区间为准，不把业务预期强加给模型。

#### 主模型

采用 CatBoostClassifier：

- 损失函数：`Logloss`，主评估指标：`Accuracy`。
- 类别接近平衡，先不设置类别权重，再通过交叉验证确认。
- 参数范围可设为 `depth=4~8`、`learning_rate=0.02~0.1`、`iterations=500~1800`、`l2_leaf_reg=3~15`。
- 使用早停并保存最佳迭代轮数。

可增加一个 LightGBM 或 HistGradientBoosting 模型。如果不同模型的折外错误具有互补性，再进行概率加权融合；否则保留单模型以降低复杂度和部署风险。

### 5.5 评估与解释

- 主指标：准确率 `ACC3`。
- 辅助指标：F1、召回率、特异度、ROC-AUC 和混淆矩阵。
- 营销场景中可重点报告高焦虑类召回率，但模型选择仍以竞赛规定的准确率为主。
- 通过 SHAP 排序识别焦虑的主要驱动因素，并绘制出行距离、充电困难度、EV 知识和电池担忧的影响曲线。
- 对接近阈值的原始评分样本进行误差分析。例如分别统计评分为 5 和 6 的样本错误率，判断边界样本是否是主要误差来源。该分析只能用于解释，不能把原始评分重新作为模型输入。

### 5.6 实施步骤

1. 根据评分生成 0/1 标签，立即从特征表删除原始评分。
2. 分层划分训练集和独立验证集。
3. 建立逻辑回归基线并检查变量方向。
4. 加入出行和充电交互特征，训练 CatBoost。
5. 进行特征消融，重点比较是否加入潜在下游变量的结果。
6. 仅基于训练集折外概率决定是否调整阈值。
7. 在独立验证集计算 `ACC3` 并开展边界样本误差分析。

## 6. 模型比较与调参策略

为避免无目的地扩大搜索空间，建议分三轮实验：

### 第一轮：基线与数据管线验证

- 使用原始特征训练逻辑回归和一个默认参数 CatBoost。
- 检查训练/验证样本数、标签分布、缺失值处理和输出标签是否合法。
- 确认模型预测结果能还原到原始样本行号。
- 对二分类任务同时报告多数类基线和 ROC-AUC。若多个不同模型的 AUC 均约为 0.5，应先判定特征来源是否缺乏信号，禁止直接扩大超参数搜索。

### 第二轮：特征消融

每次只改变一组特征，记录五折平均准确率及标准差：

- 原始特征。
- 原始特征加缺失/异常指示变量。
- 原始特征加业务交互特征。
- 严格可部署特征与扩展特征。
- 任务二单独比较 S0 严格特征、S1 真实采用意愿上界和 S2 Task1-no-home OOF 概率堆叠；最终部署只在 S0 与 S2 中选择。
- 不同类别权重。

只有平均准确率稳定提高、折间波动没有明显增大的特征组才进入下一轮。

### 第三轮：超参数优化与融合

- 使用随机搜索或 Optuna 在合理范围内搜索 30~80 组参数。
- 只有当特征组的 OOF AUC 明显高于 0.5 或准确率超过多数类基线后，才进入大规模调参。
- 二分类优化目标为 OOF 阈值准确率，同时记录固定 0.5 阈值准确率、ROC-AUC、标准差和最佳迭代数。
- 若多个参数性能接近，选择深度更小、迭代更少的模型。
- 仅当融合在折外预测上稳定优于最佳单模型时采用融合。

## 7. 可复现实现框架

推荐项目结构：

```text
project/
├─ data/
│  └─ B题数据集.csv
├─ src/
│  ├─ common.py              # 数据读取、清洗、划分和公共指标
│  ├─ train_task1.py         # 任务一训练
│  ├─ train_task2.py         # 任务二训练
│  ├─ train_task3.py         # 任务三训练
│  └─ predict.py             # 统一模型加载与预测
├─ models/                   # 模型、特征定义和标签映射
├─ outputs/                  # 验证预测、指标、图表和提交 CSV
├─ requirements.txt
└─ Introduction.pdf
```

核心实现逻辑可概括为：

```python
df = pd.read_csv(data_path, encoding="utf-8")
df["fuel_expense_invalid"] = (df["fuel_expense_per_month"] < 0).astype(int)
df.loc[df["fuel_expense_per_month"] < 0, "fuel_expense_per_month"] = np.nan

# 任务三示例：先生成标签，再删除原始评分，避免泄漏。
y = (df["range_anxiety_score"] > 5).astype(int)
X = df.drop(columns=["range_anxiety_score", "ev_adoption_likelihood"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)
```

实际代码文件头部必须按比赛规定加入 AI 辅助说明。训练时固定 Python、numpy、scikit-learn、CatBoost 等依赖版本，并记录随机种子、特征列表、参数、最佳迭代轮数和评价结果。

## 8. 结果输出与论文呈现

每个任务至少输出以下内容：

- 验证集逐行结果：`sample_id`、`y_true`、`y_pred`、各类别预测概率。
- 准确率、辅助指标和混淆矩阵。
- 交叉验证各折成绩、均值和标准差。
- 最佳模型参数、训练时长和随机种子。
- 特征重要性图、关键特征影响图和典型误判样本分析。
- 训练好的模型文件、标签映射、特征顺序及推理脚本。

建议论文中的每项任务按“问题定义—数据处理—特征工程—模型选择—调参过程—结果分析—模型解释”的顺序展开。预测结果 CSV 必须保留原始样本行号，标签格式严格使用赛题规定的 Low/Medium/High 或 0/1。

## 9. 推荐的最终方案

| 子问题 | 基线 | 推荐主模型 | 重点特征 | 关键风险 |
|---|---|---|---|---|
| 任务一 | 多项逻辑回归 | CatBoost 三分类，必要时概率融合 | 经济负担、环保与技术倾向、充电便利、认知焦虑差 | 类别不均衡、Medium/Low 识别不足 |
| 任务二 | 多数类基线 + 二元逻辑回归 | CatBoost 二分类 + Task1-no-home OOF 概率堆叠（S2）；S0 保留作对照 | 任务一 OOF 概率、收入、公共充电稀缺度、车辆与出行需求 | 跨任务标签泄漏、特征来源弱、两阶段部署复杂度 |
| 任务三 | 二元逻辑回归 | CatBoost 二分类 | 出行强度、充电困难、EV 知识、电池担忧、家充条件 | 原始评分泄漏、边界样本难分 |

总体上，仍使用 CatBoost 作为三项任务的统一主模型，以逻辑回归提供基线和可解释性，通过分层 5 折交叉验证完成特征消融与调参，并保留独立验证集进行最终检验。任务二的严格特征已被多模型诊断为弱信号来源，因此其优先工作从“继续调参”调整为“构建无泄漏的跨任务 OOF 概率特征”；只有 S2 在训练集 OOF 中稳定超过 S0 后，才增加调参和两阶段部署复杂度。
