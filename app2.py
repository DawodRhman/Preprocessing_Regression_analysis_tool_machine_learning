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

# Helper functions
def generate_data():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return pd.DataFrame(np.hstack((X, y)), columns=["X", "y"])

def preprocess_data(data):
    st.subheader("Data Preprocessing Steps")
    
    # Handle missing values
    st.write("1. Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    data = imputer.fit_transform(data)
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
    
    return pd.DataFrame(data, columns=["X", "y"])

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
    
    return theta, losses

# Streamlit App
st.title("Gradient Descent Visualizer")
st.sidebar.header("Options")

# Choose regression type
regression_type = st.sidebar.selectbox("Select Regression Type", ["Linear Regression", "Polynomial Regression"])
poly_degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2)  # Allows dynamic selection of polynomial degree

# Choose Gradient Descent type
gd_type = st.sidebar.selectbox("Select Gradient Descent Type", ["Batch Gradient Descent", "Stochastic Gradient Descent", "Mini-Batch Gradient Descent"])

# Upload data or generate random data
data_source = st.sidebar.radio("Choose Data Source", ["Generate Random Data", "Upload File"])
if data_source == "Generate Random Data":
    data = generate_data()
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
X = processed_data[["X"]].values
y = processed_data[["y"]].values

# Polynomial Regression (if selected)
if regression_type == "Polynomial Regression":
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
else:
    X_poly = X

# Visualize data
st.subheader("Data Visualization")
fig = px.scatter(x=X.flatten(), y=y.flatten(), labels={'x': 'X', 'y': 'y'}, title="Scatter Plot with Regression Line")
st.plotly_chart(fig, use_container_width=True)

# Gradient Descent Parameters
st.sidebar.subheader("Gradient Descent Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, 0.001)
iterations = st.sidebar.slider("Iterations", 1, 1000, 100)
batch_size = st.sidebar.slider("Batch Size (Mini-Batch Only)", 1, 50, 10) if gd_type == "Mini-Batch Gradient Descent" else None

# Perform Gradient Descent
st.subheader("Gradient Descent Process")
method_map = {
    "Batch Gradient Descent": "batch",
    "Stochastic Gradient Descent": "stochastic",
    "Mini-Batch Gradient Descent": "mini-batch"
}
theta, losses = gradient_descent(X_poly, y, learning_rate, iterations, batch_size, method_map[gd_type])

# Visualize Loss over Iterations
st.subheader("Loss Over Iterations")
fig = px.line(y=losses, x=range(iterations), labels={'x': 'Iterations', 'y': 'Loss'}, title="Gradient Descent Loss")
st.plotly_chart(fig, use_container_width=True)

# Visualize the Fitted Model
st.subheader("Fitted Model Visualization")
fig = px.scatter(x=X.flatten(), y=y.flatten(), labels={'x': 'X', 'y': 'y'}, title="Data and Fitted Model")
fig.add_scatter(x=X.flatten(), y=X_poly.dot(theta).flatten(), mode='lines', name='Fitted Line')
st.plotly_chart(fig, use_container_width=True)

# Model Evaluation (R-squared)
r2 = r2_score(y, X_poly.dot(theta))
st.subheader("Model Evaluation")
st.write(f"R-squared: {r2:.4f}")

# Final Model Parameters
st.subheader("Final Model Parameters")
st.write(f"Theta: {theta.flatten()}")

