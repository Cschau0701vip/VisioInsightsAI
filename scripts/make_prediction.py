import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Step 1: Simulate Data (Replace this with real data for better accuracy)
np.random.seed(42)
data_size = 1000
# Initialize the dataframe globally
df = None

# The uploaded file's data is already stored in 'df'
# Remove the redundant line
# data = {
#   'year': np.random.randint(2000, 2023, size=data_size),
#   'mileage': np.random.randint(10000, 200000, size=data_size),
#   'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], size=data_size),
#   'resale_value': np.random.normal(20000, 5000, size=data_size)  # More realistic resale values
# }

def process_and_visualize_data(uploaded_file, output_dir, datetime, selected_graph_type):
  global df  # Use the global df variable to store the data
  print("process_and_visualize_data")
  # Create the output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)
  # Check the file type and read the data accordingly
  try:
    print("reading file: - " + uploaded_file)
    if uploaded_file.endswith('.csv'):
      print("csv")
      df = pd.read_csv(uploaded_file)
      if df.empty:
        st.error("The uploaded CSV file is empty. Please upload a valid file.")
        return
    elif uploaded_file.endswith('.json'):
      print("json")
      df = pd.read_json(uploaded_file)
      if df.empty:
        st.error("The uploaded JSON file is empty. Please upload a valid file.")
        return
    else:
      st.error("Unsupported file format. Please upload a CSV or JSON file.")
      return
  except ValueError as e:
    st.error(f"Error reading the file: {e}")
    return
  
  # Convert categorical data to numerical using one-hot encoding
  df = pd.get_dummies(df, columns=['condition'], drop_first=True)

  # Features and target variable
  X = df.drop('resale_value', axis=1)
  y = df['resale_value']

  # Step 2: Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train the model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Step 3: Make predictions
  y_pred = model.predict(X_test)

  # Step 4: Evaluate the model
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  print(f'Mean Squared Error: {mse}')
  print(f'R^2 Score: {r2}')

  # Call the functions
  match selected_graph_type:
    case 'scatter':
      save_scatter_plot(y_test, y_pred, X_test, output_dir, datetime)
    case 'residual':
      residuals = y_test - y_pred
      save_residual_plot(residuals, output_dir, datetime)
    case 'correlation':
      save_correlation_heatmap(df, output_dir, datetime)
    case 'linear':
      save_linear_plot(X_test, y_pred, output_dir, datetime)
    case 'pie':
      save_pie_chart(df, output_dir, datetime)
    case _:
      save_all_plots(y_test, y_pred, X_test, output_dir, datetime)

def save_all_plots(y_test, y_pred, X_test, output_dir, timestamp):
  print("save_all_plots")
  save_scatter_plot(y_test, y_pred, X_test, output_dir, timestamp)
  residuals = y_test - y_pred
  save_residual_plot(residuals, output_dir, timestamp)
  save_correlation_heatmap(df, output_dir, timestamp)
  save_linear_plot(X_test, y_pred, output_dir, timestamp)
  save_pie_chart(df, output_dir)

def save_scatter_plot(y_test, y_pred, X_test, output_dir, timestamp):
  print("save_scatter_plot")
  plt.figure(figsize=(12, 8))
  sns.scatterplot(x=y_test, y=y_pred, hue=X_test['year'], palette='viridis', alpha=0.7, edgecolor='k')
  plt.plot([0, 30000], [0, 30000], color='red', linestyle='--', label='Perfect Prediction')
  plt.title('Actual vs Predicted Resale Values')
  plt.xlabel('Actual Resale Value ($)')
  plt.ylabel('Predicted Resale Value ($)')
  plt.legend(title='Year')
  plt.grid(alpha=0.5)
  plt.savefig(os.path.join(output_dir, f'scatter_actual_vs_predicted_resale_values_{timestamp}.png'))
  # plt.show()

def save_residual_plot(residuals, output_dir, timestamp):
  print("save_residual_plot")
  plt.figure(figsize=(12, 8))
  sns.histplot(residuals, kde=True, bins=30, color='blue')
  plt.title('Residuals Distribution')
  plt.xlabel('Residuals (Actual - Predicted)')
  plt.ylabel('Frequency')
  plt.grid(alpha=0.5)
  plt.savefig(os.path.join(output_dir, f'residuals_distribution_{timestamp}.png'))
  # plt.show()

def save_correlation_heatmap(df, output_dir, timestamp):
  print("save_correlation_heatmap")
  plt.figure(figsize=(10, 6))
  correlation_matrix = df.corr()
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
  plt.title('Correlation Matrix of Features')
  plt.savefig(os.path.join(output_dir, f'correlation_matrix_{timestamp}.png'))
  # plt.show()

def save_linear_plot(X_test, y_pred, output_dir, timestamp):
  print("save_linear_plot")
  plt.figure(figsize=(12, 8))
  plt.plot(X_test['mileage'], y_pred, 'o', color='green', alpha=0.5, label='Predicted Resale Value')
  plt.title('Mileage vs Predicted Resale Value')
  plt.xlabel('Mileage')
  plt.ylabel('Predicted Resale Value ($)')
  plt.grid(alpha=0.5)
  plt.legend()
  plt.savefig(os.path.join(output_dir, f'linear_mileage_vs_predicted_resale_value_{timestamp}.png'))
  # plt.show()

def save_pie_chart(df, output_dir, timestamp):
  print("save_pie_chart")
  # Check if 'condition' column exists, if not, add it
  if 'condition' not in df.columns:
    df['condition'] = np.random.choice(['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'], size=len(df))
  condition_counts = pd.Series(df['condition']).value_counts()
  plt.figure(figsize=(8, 8))
  plt.pie(condition_counts, labels=condition_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
  plt.title('Distribution of Vehicle Conditions')
  plt.savefig(os.path.join(output_dir, f'pie_condition_distribution_{timestamp}.png'))
  # plt.show()
