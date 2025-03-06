import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import plotly.express as px
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Helper functions
def generate_data(features=1, samples=100):
    np.random.seed(42)
    X = 2 * np.random.rand(samples, features)
    coeffs = np.random.rand(features, 1) * 4
    y = 4 + X.dot(coeffs) + np.random.randn(samples, 1)
    columns = [f"X{i+1}" for i in range(features)]
    return pd.DataFrame(np.hstack((X, y)), columns=columns + ["y"])

def preprocess_data(data):
    st.subheader("Data Preprocessing Steps")

    # Save column names
    column_names = data.columns

    # Handle missing values
    st.write("1. Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    try:
        data = imputer.fit_transform(data)
    except Exception as e:
        st.error("Error during missing value handling: Non-numeric data detected.")
        st.stop()
    st.write(" - Missing values handled.")

    # Normalize the data
    st.write("2. Normalizing the data...")
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    st.write(" - Data normalized.")

    # Outlier removal
    st.write("3. Removing outliers...")
    data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
    st.write(" - Outliers removed.")

    if len(data) < 10:
        st.error("Insufficient data points after preprocessing. Ensure the dataset has enough samples.")
        st.stop()

    # Convert back to DataFrame with original column names
    return pd.DataFrame(data, columns=column_names)

def plot_data(X, y, title):
    plt.scatter(X, y, color="blue")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    st.pyplot(plt.gcf())

def gradient_descent(X, y, lr, iterations, batch_size=None, method="batch"):
    m, n = X.shape
    theta = np.random.randn(n, 1)  # Random initialization
    losses = []
    theta_history = [theta.copy()]  # To store theta updates for visualization
    progress = st.progress(0)  # Real-time progress bar

    for iteration in range(iterations):
        if method == "batch":
            gradients = 2/m * X.T.dot(X.dot(theta) - y)
        elif method == "stochastic":
            i = np.random.randint(m)
            xi = X[i:i+1]
            yi = y[i:i+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        elif method == "mini-batch":
            indices = np.random.randint(0, m, batch_size)
            xi = X[indices]
            yi = y[indices]
            gradients = 2/batch_size * xi.T.dot(xi.dot(theta) - yi)

        theta -= lr * gradients
        loss = np.mean((X.dot(theta) - y) ** 2)
        losses.append(loss)
        theta_history.append(theta.copy())

        progress.progress((iteration + 1) / iterations)

    progress.empty()
    return theta, losses, theta_history

# Streamlit App
st.title("Gradient Descent Visualizer")
st.sidebar.header("Options")

# Instructions
st.sidebar.markdown(
    """
    **How to Use the App:**
    - Choose a regression type (Linear/Polynomial).
    - Configure gradient descent parameters in the sidebar.
    - Upload your dataset or generate random data.
    - Visualize and evaluate the results interactively.
    """
)

# Tooltips for Parameters
def tooltip(label, help_text):
    return st.sidebar.slider(label, 0.001, 1.0, 0.1, 0.001, help=help_text)

# Choose regression type
regression_type = st.sidebar.selectbox("Select Regression Type", ["Linear Regression", "Polynomial Regression"])
poly_degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2, help="Degree of polynomial for regression.")

# Choose Gradient Descent type
gd_type = st.sidebar.selectbox(
    "Select Gradient Descent Type", 
    ["Batch Gradient Descent", "Stochastic Gradient Descent", "Mini-Batch Gradient Descent"]
)

# Adjust dynamic batch size slider
if gd_type == "Mini-Batch Gradient Descent":
    batch_size = st.sidebar.slider("Batch Size", 1, 50, 10, help="Mini-batch size for gradient descent.")
else:
    batch_size = None

# Upload data or generate random data
data_source = st.sidebar.radio("Choose Data Source", ["Generate Random Data", "Upload File"])
if data_source == "Generate Random Data":
    num_features = st.sidebar.slider("Number of Features", 1, 10, 1, help="Number of features for generated data.")
    num_samples = st.sidebar.slider("Number of Samples", 10, 1000, 100, help="Number of samples for generated data.")
    data = generate_data(features=num_features, samples=num_samples)
    st.subheader("Generated Data")
    st.dataframe(data)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your file (CSV/XLS)", type=["csv", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Failed to read the file. Error: {e}")
            st.stop()
    else:
        st.warning("Please upload a file to proceed.")
        st.stop()

# Preprocess data
processed_data = preprocess_data(data)
X = processed_data.iloc[:, :-1].values
y = processed_data.iloc[:, -1:].values

# Split data into training and test sets
split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8, help="Proportion of data for training.")
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Polynomial Regression (if selected)
if regression_type == "Polynomial Regression":
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
else:
    X_train_poly, X_test_poly = X_train, X_test

# Gradient Descent Parameters
st.sidebar.subheader("Gradient Descent Parameters")
learning_rate = tooltip("Learning Rate", "Controls the step size for updates.")
iterations = st.sidebar.slider("Iterations", 1, 1000, 100, help="Number of gradient descent steps.")

# Perform Gradient Descent
st.subheader("Gradient Descent Process")
method_map = {
    "Batch Gradient Descent": "batch",
    "Stochastic Gradient Descent": "stochastic",
    "Mini-Batch Gradient Descent": "mini-batch"
}

# Train model using gradient descent
theta, losses, theta_history = gradient_descent(X_train_poly, y_train, learning_rate, iterations, batch_size, method_map[gd_type])

# Visualize Loss over Iterations
st.subheader("Loss Over Iterations")
fig = px.line(y=losses, x=range(len(losses)), labels={'x': 'Iterations', 'y': 'Loss'}, title="Gradient Descent Loss")
st.plotly_chart(fig, use_container_width=True)

# Visualize Gradient Descent in 3D (if 2D data)
if X_train_poly.shape[1] <= 3:  # Only visualize if data is 2D or 1D (plus bias term)
    st.subheader("3D Gradient Descent Visualization")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    theta_0_vals = [t[0, 0] for t in theta_history]
    theta_1_vals = [t[1, 0] if t.shape[0] > 1 else 0 for t in theta_history]  # Handle cases with single feature
    loss_vals = losses

    # Ensure all lists have the same length
    min_len = min(len(theta_0_vals), len(theta_1_vals), len(loss_vals))
    theta_0_vals = theta_0_vals[:min_len]
    theta_1_vals = theta_1_vals[:min_len]
    loss_vals = loss_vals[:min_len]

    ax.plot(theta_0_vals, theta_1_vals, loss_vals, marker='o', label='Gradient Descent Path')
    ax.set_title("Gradient Descent Path")
    ax.set_xlabel("Theta 0")
    ax.set_ylabel("Theta 1")
    ax.set_zlabel("Loss")
    st.pyplot(fig)

# Evaluate model on test data
st.subheader("Model Evaluation")
y_pred = X_test_poly.dot(theta)
r2 = r2_score(y_test, y_pred)
st.write(f"R-squared on Test Set: {r2:.4f}")

# Visualize the Fitted Model
st.subheader("Fitted Model Visualization")
fig = px.scatter(x=X[:, 0].flatten(), y=y.flatten(), labels={'x': 'X', 'y': 'y'}, title="Data and Fitted Model")
fig.add_scatter(x=X[:, 0].flatten(), y=np.vstack((X_train_poly, X_test_poly)).dot(theta).flatten(), mode='lines', name='Fitted Line')
st.plotly_chart(fig, use_container_width=True)

# Final Model Parameters
st.subheader("Final Model Parameters")
st.write(f"Theta: {theta.flatten()}")
