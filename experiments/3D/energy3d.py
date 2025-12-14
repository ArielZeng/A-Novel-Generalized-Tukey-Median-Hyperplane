import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy.linalg import null_space
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# Function to compute a perpendicular vector to a given PCA component
def compute_perpendicular_vector(vector):
    # Find the null space of the vector to get a perpendicular vector
    null_vec = null_space(vector.reshape(1, -1))
    # Return the first basis vector of the null space as a perpendicular vector
    return null_vec[:, 0]  # Perpendicular to the input vector

# Function to split data based on Tukey Median and PCA
def split_data_based_on_tukey_median(data, overall_tukey_median, pca_direction):
    perp_dir = compute_perpendicular_vector(pca_direction)
    return np.dot(data - overall_tukey_median, perp_dir) < 0  # Split the data based on perpendicular direction

# Function to fit a hyperplane based on Tukey Median values using SVD
def fit_hyperplane(tukey_median_values):
    tukey_median_values = np.array(tukey_median_values)

    # 确保矩阵是方阵才能计算行列式
    if tukey_median_values.shape[0] != tukey_median_values.shape[1]:
        print("⚠ Warning: Non-square matrix encountered! Skipping this hyperplane.")
        return None, None

    try:
        determinant = np.linalg.det(tukey_median_values)
    except np.linalg.LinAlgError:
        print("⚠ Warning: Singular matrix encountered! Skipping this hyperplane.")
        return None, None

    if np.abs(determinant) < 1e-10:  # 设置阈值，防止数值误差导致误判
        print("⚠ Warning: det(A) ≈ 0, matrix is singular! Skipping this hyperplane.")
        return None, None

    # 进行 SVD 分解以求超平面法向量
    _, _, Vt = np.linalg.svd(tukey_median_values)

    # 取最后一行作为超平面法向量
    normal_vector = Vt[-1, :]

    # 计算截距
    intercept = np.dot(normal_vector, tukey_median_values.mean(axis=0))

    print("Hyperplane normal vector (coefficients):", normal_vector)
    print("Hyperplane intercept:", intercept)

    return normal_vector, intercept

# Main function to process data with Tukey and PCA
def process_data_with_tukey(data_train, data_test, feature_columns, target_column):
    print("\n=== Step 1: Compute Overall Tukey Median ===")
    overall_tukey_median = Tukey_Median(data_train)
    print(f'Overall Tukey Median: {overall_tukey_median}')

    print("\n=== Step 2: Perform First PCA and Split Data ===")
    pca1 = PCA(n_components=1)
    pca1.fit(data_train)  # ✅ 只对 feature_columns 进行 PCA

    primary_direction1 = pca1.components_[0]

    mask1 = split_data_based_on_tukey_median(data_train.values, overall_tukey_median, primary_direction1)
    left_data1 = data_train[mask1]
    right_data1 = data_train[~mask1]

    print(f"Left Data1 Shape: {left_data1.shape}, Right Data1 Shape: {right_data1.shape}")

    tukey_median_left1 = Tukey_Median(left_data1)
    tukey_median_right1 = Tukey_Median(right_data1)

    print(f'Tukey Median for first split left: {tukey_median_left1}')
    print(f'Tukey Median for first split right: {tukey_median_right1}')

    # Step 5: Fit a hyperplane using the overall and split Tukey Medians
    tukey_medians = [overall_tukey_median, tukey_median_left1, tukey_median_right1]
    normal_vector, intercept = fit_hyperplane(tukey_medians)

    # Step 6: Calculate distances from each test point to the hyperplane
    X_test = data_test[feature_columns].values  # Use specified features
    numerator = np.abs(np.dot(X_test, normal_vector[:len(feature_columns)]) + intercept)
    denominator = np.linalg.norm(normal_vector[:len(feature_columns)])
    distances = numerator / denominator  # This gives the distance of each test point to the hyperplane

    # Use these distances to compute MAE and R²
    mae_test = np.mean(distances)  # Average distance as a measure of absolute fit
    r2_test = 1 - (np.var(distances) / np.var(
        np.abs(data_test[target_column].values - np.mean(data_test[target_column].values))))

    return {
        "tukey median mae_test": mae_test,
        "tukey median r2_test": r2_test
    }

# Define columns
feature_columns = ['Square Footage', 'Number of Occupants']  # Selected features for fitting the hyperplane
target_column = 'Energy Consumption'  # Target variable for distance-based R² calculation

# Generate a regression dataset with 6 features
data = pd.read_csv(r"D:\course\P\Project4\pythonProject\3\test_energy_data.csv")

data_selected = data[['Square Footage', 'Number of Occupants', 'Energy Consumption']]
# Split the data into training and testing sets
data_train, data_test = train_test_split(data_selected, test_size=0.3, random_state=0)

# Process the training and testing datasets (assuming process_data_with_tukey is defined)
result = process_data_with_tukey(data_train, data_test, feature_columns, target_column)
print(result)
linear_model = LinearRegression()
linear_model.fit(data_train[feature_columns], data_train[target_column])
y_pred = linear_model.predict(data_test[feature_columns])

# Evaluate the model using MAE and R2
mae = mean_absolute_error(data_test[target_column], y_pred)
r2 = r2_score(data_test[target_column], y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")