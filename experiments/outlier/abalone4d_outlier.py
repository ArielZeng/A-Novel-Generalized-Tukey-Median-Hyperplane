import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import train_test_split
from scipy.linalg import lstsq, null_space

# Activate automatic conversion between pandas and R dataframes
pandas2ri.activate()

# Import necessary R packages
TukeyRegion = importr('TukeyRegion')

# Function to compute Tukey Median using R's TukeyRegion package
def Tukey_Median(data):
    # Convert the pandas DataFrame to an R DataFrame
    data_r = pandas2ri.py2rpy(data)
    robjects.globalenv['x'] = data_r  # Assign to global variable 'x'

    # R code to apply TukeyMedian and get the summary
    r_code = '''
    library(TukeyRegion)
    matrix_data <- as.matrix(x)
    Tm <- TukeyMedian(matrix_data)
    Tm$barycenter  
    '''

    # Execute the R code and store the result
    barycenter = robjects.r(r_code)

    # Return the barycenter (Tukey Median)
    return [float(x) for x in barycenter]

def compute_tukey_depth(data):
    """
    Compute the Tukey depth for each point in the dataset.
    Returns a DataFrame with points and their corresponding depths.
    """

    # Convert pandas DataFrame to R matrix
    data_r = pandas2ri.py2rpy(data)
    robjects.globalenv['x'] = data_r  # Assign to global variable 'x'

    # R code to calculate Tukey depth using depth.halfspace function
    r_code = '''
    library(TukeyRegion)
    matrix_data <- as.matrix(x)
    depths <- depth.halfspace(matrix_data, matrix_data)
    depths
    '''

    # Execute the R code and get Tukey depths
    depths = np.array(robjects.r(r_code))
    depth_df = pd.DataFrame({'Depth': depths}, index=data.index)  # 防止索引错位

    return depth_df

# Function to compute Tukey depth

# Function to get the top N points with the highest Tukey depth
def get_top_tukey_depth_points(data, top_n=None):
    """
    Get the top N points with the highest Tukey depth.
    If top_n is None, return all points sorted by Tukey depth.
    """
    depth_df = compute_tukey_depth(data)

    # Combine original data with depths
    data_with_depth = data.copy()
    data_with_depth['Depth'] = depth_df['Depth']

    # Sort by depth in descending order
    data_with_depth = data_with_depth.sort_values(by='Depth', ascending=False)

    # If top_n is None, return the full sorted dataset
    if top_n is None:
        return data_with_depth
    else:
        return data_with_depth.head(top_n)

# ===== 计算 PCA 方向 =====
def compute_pca_direction(data, feature_columns):
    pca = PCA(n_components=1)
    pca.fit(data[feature_columns])
    return pca.components_[0]  # 主方向向量


def compute_perpendicular_vector(vector):
    # Find the null space of the vector to get a perpendicular vector
    null_vec = null_space(vector.reshape(1, -1))
    # Return the first basis vector of the null space as a perpendicular vector
    return null_vec[:, 0]  # Perpendicular to the input vector

# Function to split data based on Tukey Median and PCA
def split_data_based_on_tukey_median(data, overall_tukey_median, pca_direction):
    perp_dir = compute_perpendicular_vector(pca_direction)
    return np.dot(data - overall_tukey_median, perp_dir) < 0  # Split the data based on perpendicular direction

def fit_hyperplane(tukey_median_values):
    tukey_median_values = np.array(tukey_median_values)
    num_points, num_features = tukey_median_values.shape

    if num_points < num_features:
        print("⚠ Warning: Not enough points for hyperplane fitting! Skipping.")
        return None, None

    _, _, Vt = np.linalg.svd(tukey_median_values)
    normal_vector = Vt[-1, :]
    intercept = -np.dot(normal_vector, tukey_median_values.mean(axis=0))
    return normal_vector, intercept

# **最终方法：结合 Tukey Median 和 PCA 方向选点**
def select_tukey_pca_combined_points(data, feature_columns, target_column, alpha=0.5):
    """
    选择 Tukey Median 以及 PCA 方向最相关的点，作为最终用于回归的点集。

    参数:
    - data: pandas DataFrame, 包含所有数据
    - feature_columns: list, 需要用于建模的特征列
    - target_column: str, 目标变量列
    - alpha: float, Tukey 深度和 PCA 方向的权重 (0.5 = 平衡 Tukey 深度和 PCA 方向)

    返回:
    - selected_points: pandas DataFrame, 最终选择的点集合
    """

    dim = data.shape[1]  # ✅ 计算整个数据集的维度（包含 target_column）
    num_needed = dim - 3  # 还需要多少个点（PCA 方向选取），因为已经选了 3 个 Tukey Medians

    ## **Step 1: 计算 overall Tukey Median**
    overall_tukey_median = np.array(Tukey_Median(data))  # ✅ 保持完整维度（包含 `target_column`）
    tukey_median_point = pd.DataFrame([overall_tukey_median], columns=feature_columns + [target_column])
    print(f"✅ Overall Tukey Median: {overall_tukey_median}")

    ## **Step 2: 计算所有点的 Tukey 深度**
    top_tukey_points = get_top_tukey_depth_points(data)  # 所有点按 Tukey 深度排序
    top_tukey_points["Depth Rank"] = range(1, len(top_tukey_points) + 1)

    ## **Step 3: PCA 分割数据**
    pca_direction = compute_pca_direction(data, feature_columns)

    feature_data = data[feature_columns].values  # 只提取 `feature_columns` 计算 PCA 分割
    mask = split_data_based_on_tukey_median(feature_data, overall_tukey_median[:-1], pca_direction)

    left_data = data[mask]  # **完整数据（包含 `target_column`）**
    right_data = data[~mask]  # **完整数据（包含 `target_column`）**

    ## **Step 4: 计算 Tukey Median（左右分割数据）**
    tukey_median_left = np.array(Tukey_Median(left_data))  # **包含 `target_column`**
    tukey_median_right = np.array(Tukey_Median(right_data))  # **包含 `target_column`**

    print(f"✅ Tukey Median Left: {tukey_median_left}")
    print(f"✅ Tukey Median Right: {tukey_median_right}")

    # **转换成 DataFrame**
    tukey_median_left_df = pd.DataFrame([tukey_median_left], columns=feature_columns + [target_column])
    tukey_median_right_df = pd.DataFrame([tukey_median_right], columns=feature_columns + [target_column])

    ## **Step 5: 计算 PCA 方向**
    top_tukey_points["PCA Projection"] = top_tukey_points[feature_columns].dot(pca_direction)

    ## **Step 6: 归一化 PCA 投影**
    min_proj, max_proj = top_tukey_points["PCA Projection"].min(), top_tukey_points["PCA Projection"].max()
    top_tukey_points["PCA Projection Norm"] = (top_tukey_points["PCA Projection"] - min_proj) / (max_proj - min_proj)

    ## **Step 7: 计算综合得分（Tukey 深度 + PCA 方向投影）**
    scores = alpha * top_tukey_points["Depth"] + (1 - alpha) * top_tukey_points["PCA Projection Norm"]
    sorted_points = top_tukey_points.iloc[np.argsort(scores)[::-1]]  # 排序得分

    # **Step 8: 选取 top_n 个点**
    selected_pca_points = sorted_points.head(num_needed)

    ## **Step 9: 组合最终点集**
    selected_points = pd.concat([tukey_median_point, tukey_median_left_df, tukey_median_right_df, selected_pca_points])
    print(f"✅ Selected {len(selected_points)} points (3 Tukey Medians + {num_needed} PCA-aligned points)")
    print(f"✅ Overall Tukey Median: \n{tukey_median_point}")
    print(f"✅ PCA-aligned points: \n{selected_pca_points}")

    return selected_points


# **主函数**
def process_data_with_tukey(data_train, data_test, feature_columns, target_column):
    selected_points = select_tukey_pca_combined_points(data_train, feature_columns, target_column)

    # **拟合 Tukey-based 超平面**
    X_selected = selected_points[feature_columns].values
    y_selected = selected_points[target_column].values
    tukey_median_values = np.column_stack((X_selected, y_selected))
    normal_vector, intercept = fit_hyperplane(tukey_median_values)

    if normal_vector is None:
        return {"mae_test": None, "r2_test": None}

    # **计算误差**
    X_test = data_test[feature_columns].values
    numerator = np.abs(np.dot(X_test, normal_vector[:-1]) + intercept)
    denominator = np.linalg.norm(normal_vector[:-1])
    distances = numerator / denominator
    mae_test = np.mean(distances)
    r2_test = 1 - (np.var(distances) / np.var(data_test[target_column].values))

    return {"mae_test": mae_test, "r2_test": r2_test}


def add_outliers(data, feature_columns, target_column, outlier_ratio, seed=None):
    """
    Add outliers using IQR rule to ensure outliers fall outside normal range.

    Parameters:
        data: pandas DataFrame
        feature_columns: list, feature columns
        target_column: str, target column
        outlier_ratio: float, proportion of outliers
        seed: int, random seed (optional)

    Returns:
        pandas DataFrame with outliers
    """
    if seed is not None:
        np.random.seed(seed)  # ✅ 确保随机性可复现

    data_with_outliers = data.copy()
    n_outliers = int(len(data) * outlier_ratio)

    # ✅ 确保 `replace=False` 只在 `np.random.choice()` 内部使用，而不传递到 `add_outliers()`
    outlier_indices = np.random.choice(data.index, n_outliers, replace=False)

    for col in feature_columns:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1

        # ✅ 确保异常值落在 lower_bound 之外 或 upper_bound 之外
        outlier_values = np.random.uniform(
            low=Q1 - 1.5 * IQR,  # ✅ 让 lower bound 的异常值更小
            high=Q3 + 1.5 * IQR,  # ✅ 让 upper bound 的异常值更大
            size=n_outliers
        )

        # ✅ 赋值异常值
        data_with_outliers.loc[outlier_indices, col] = outlier_values

    # ✅ 处理 target_column 异常值（确保整数）
    Q1_target = np.percentile(data[target_column], 25)
    Q3_target = np.percentile(data[target_column], 75)
    IQR_target = Q3_target - Q1_target

    lower_target = Q1_target - 1.5 * IQR_target
    upper_target = Q3_target + 1.5 * IQR_target

    # ✅ 生成整数异常目标值
    target_outlier_values = np.random.uniform(
        low=lower_target, high=upper_target, size=n_outliers
    )

    # ✅ 确保 target_column 仍然是整数
    data_with_outliers.loc[outlier_indices, target_column] = np.round(target_outlier_values).astype(int)

    return data_with_outliers



# ===== 数据集 =====
feature_columns = ['Length', 'Diameter', 'Whole weight']
target_column = 'Rings'

data = pd.read_csv(r"D:\course\P\Project4\pythonProject\3\abalone.csv")
data_selected = data[feature_columns + [target_column]]
RANDOM_SEED = 0
data_sampled = data_selected.sample(n=100, random_state=RANDOM_SEED)
# **Step 1: First, split data into train and test sets**
data_train, data_test = train_test_split(data_sampled, test_size=0.3, random_state=RANDOM_SEED)

outlier_ratio = 0.3  # 5%, 10%, 20%, 30% outliers
print(f"\n=== Running Experiment with {int(outlier_ratio * 100)}% Outliers ===")

try:
    # **Step 2: 向训练数据集添加 Outliers**
    np.random.seed(RANDOM_SEED)  # ✅ 确保可复现性
    data_train_with_outliers = add_outliers(data_train, feature_columns, target_column, outlier_ratio=outlier_ratio, seed=RANDOM_SEED)

    # **Step 3: 用 Tukey Median + PCA 选择点**
    selected_points = select_tukey_pca_combined_points(data_train_with_outliers, feature_columns, target_column)

    # **Step 4: 拟合 Tukey-based 超平面**
    X_selected = selected_points[feature_columns].values
    y_selected = selected_points[target_column].values
    tukey_median_values = np.column_stack((X_selected, y_selected))

    normal_vector, intercept = fit_hyperplane(tukey_median_values)

    if normal_vector is None:
        print(f"⚠ Skipping {int(outlier_ratio * 100)}% Outliers: Tukey-based model failed.")
    else:
        # **Step 5: 计算 Tukey-based 回归的距离**
        X_test = data_test[feature_columns].values
        y_test = data_test[target_column].values

        numerator = np.abs(np.dot(X_test, normal_vector[:-1]) + intercept)
        denominator = np.linalg.norm(normal_vector[:-1])
        distances = numerator / denominator

        mae_tukey = np.mean(distances)
        r2_tukey = 1 - (np.var(distances) / np.var(np.abs(y_test - np.mean(y_test))))

        # **Step 6: 训练和评估线性回归**
        linear_model = LinearRegression()
        linear_model.fit(data_train_with_outliers[feature_columns], data_train_with_outliers[target_column])
        y_pred_lr = linear_model.predict(X_test)

        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)

        # **打印结果**
        print(f"\n📊 Results for {int(outlier_ratio * 100)}% Outliers:")
        print(f"Tukey-based Fit -> MAE: {mae_tukey:.4f}, R²: {r2_tukey:.4f}")
        print(f"Linear Regression -> MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}")

except Exception as e:
    print(f"❌ Error encountered for {int(outlier_ratio * 100)}% Outliers: {e}")