"""
California Housing Price Prediction Demo
======================================

This script demonstrates a complete machine learning pipeline for predicting
housing prices in California using the California Housing dataset.
The script can be converted to a Jupyter notebook using:
    jupyter nbconvert --to notebook --execute housing_price_prediction.py
"""

# %% [markdown]
# 1. Setup and Dependencies
# ------------------------
# First, let's import all the necessary libraries.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from IPython.display import display, Markdown

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# 2. Load and Explore the Data
# ---------------------------
# We'll use the California Housing dataset, which contains information about
# housing districts in California from the 1990 Census.

# %%
# Load the dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target * 100000  # Convert to actual dollar values

# Display basic information about the dataset
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe().to_string())

# %% [markdown]
# 3. Data Visualization
# --------------------
# Let's visualize the data to understand the distributions and relationships.

# %%
# Set up the figure and axes
plt.figure(figsize=(15, 10))

# Define units for each feature
units = {
    'MedInc': ' ($10,000s)',
    'HouseAge': ' (years)',
    'AveRooms': ' (rooms)',
    'AveBedrms': ' (bedrooms)',
    'Population': ' (people)',
    'AveOccup': ' (people/room)',
    'Latitude': ' (degrees)',
    'Longitude': ' (degrees)',
    'MedHouseVal': ' ($)'
}

# Plot histograms for all features
plt.figure(figsize=(15, 12))
for i, col in enumerate(df.columns, 1):
    plt.subplot(3, 3, i)
    
    # Apply log scale for specific columns with wide ranges
    if col in ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']:
        plt.xscale('log')
        
    sns.histplot(df[col], kde=True, bins=30)
    
    # Add units to x-axis label
    unit = units.get(col, '')
    plt.xlabel(f'{col}{unit}')
    plt.title(f'Distribution of {col}')
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    
plt.tight_layout()
plt.suptitle('Feature Distributions', y=1.02, fontsize=14, fontweight='bold')
plt.subplots_adjust(top=0.92)
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, 
            fmt=".2f", annot_kws={"size": 10})

# Improve readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Correlation Matrix of Housing Features', pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# 4. Data Preprocessing
# --------------------
# Prepare the data for modeling by handling missing values, encoding
# categorical variables, and scaling features.

# %%
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Split features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# %% [markdown]
# 5. Model Training
# ----------------
# We'll train and evaluate multiple regression models to predict housing prices.

# %%
# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
}

# Dictionary to store model performance
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Store results
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_rmse': cv_rmse
    }
    
    # Print results
    print(f"{name} Performance:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  CV RMSE: ${cv_rmse:,.2f}")

# %% [markdown]
# 6. Model Comparison
# ------------------
# Let's compare the performance of all models.

# %%
# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['rmse'] for m in results],
    'MAE': [results[m]['mae'] for m in results],
    'R²': [results[m]['r2'] for m in results],
    'CV RMSE': [results[m]['cv_rmse'] for m in results]
}).sort_values('RMSE')

print("\nModel Comparison:")
print(results_df.to_string())

# Plot model performance
plt.figure(figsize=(14, 6))
metrics = ['RMSE', 'MAE', 'R²']

for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    ax = sns.barplot(x='Model', y=metric, data=results_df, palette='viridis')
    
    # Add values on top of bars
    for p in ax.patches:
        if metric == 'R²':
            ax.annotate(f"{p.get_height():.3f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 9), 
                       textcoords='offset points')
        else:
            ax.annotate(f"${p.get_height():,.0f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 9), 
                       textcoords='offset points')
    
    # Format y-axis for R² differently
    if metric == 'R²':
        plt.ylim(0, 1.1)
        plt.ylabel('R² Score')
    else:
        # Add dollar sign and format y-axis for monetary values
        ax.yaxis.set_major_formatter('${x:,.0f}')
        plt.ylabel(f'{metric} (in $)')
    
    plt.title(f'Model {metric} Comparison', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.suptitle('Model Performance Comparison', y=1.05, fontsize=14, fontweight='bold')
plt.subplots_adjust(top=0.85)
plt.show()

# %% [markdown]
# 7. Feature Importance
# --------------------
# Let's analyze which features are most important in our best model.

# %%
# Get the best model
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

# Plot feature importance
if hasattr(best_model, 'feature_importances_'):
    # For tree-based models
    feature_importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)  # Sort ascending for horizontal bar plot
    
    plt.figure(figsize=(12, 6))
    
    # Create a horizontal bar plot
    ax = sns.barplot(x=feature_importance.values, y=feature_importance.index, 
                    palette='viridis_r', orient='h')
    
    # Add value labels
    for i, v in enumerate(feature_importance):
        ax.text(v + 0.01, i, f"{v:.3f}", color='black', va='center')
    
    # Add units to x-axis if available
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'Feature Importance - {best_model_name}', fontweight='bold', pad=15)
    
    # Add grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
else:
    # For linear regression
    if hasattr(best_model, 'coef_'):
        coefficients = pd.Series(
            best_model.coef_,
            index=X.columns
        ).sort_values(ascending=True)  # Sort ascending for horizontal bar plot
        
        plt.figure(figsize=(12, 6))
        
        # Create a horizontal bar plot
        ax = sns.barplot(x=coefficients.values, y=coefficients.index, 
                        palette='coolwarm', orient='h')
        
        # Add value labels
        for i, v in enumerate(coefficients):
            ax.text(v, i, f"{v:,.2f}", color='black', va='center')
        
        # Add a vertical line at zero
        plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
        
        # Add units to x-axis
        plt.xlabel('Coefficient Value (impact on house price in $)')
        plt.ylabel('Features')
        plt.title(f'Feature Coefficients - {best_model_name}', fontweight='bold', pad=15)
        
        # Add grid for better readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# 8. Make Predictions
# -------------------
# Let's use our best model to make some example predictions.

# %%
# Select a few random samples from the test set
# Ensure we don't go out of bounds
n_samples = min(5, len(X_test))
sample_indices = np.random.choice(range(len(X_test)), size=n_samples, replace=False)
samples = X_test.iloc[sample_indices].copy()
samples_scaled = X_test_scaled[sample_indices]

# Make predictions
predictions = best_model.predict(samples_scaled)

# Create a DataFrame to display the results
results = pd.DataFrame({
    'Actual': y_test.iloc[sample_indices].values,
    'Predicted': predictions,
    'Difference': y_test.iloc[sample_indices].values - predictions
})

# Format the DataFrame for better display
results_formatted = results.copy()
for col in ['Actual', 'Predicted', 'Difference']:
    results_formatted[col] = results_formatted[col].apply(lambda x: f"${x:,.2f}")

print("\nExample Predictions:")
print(pd.concat([samples.reset_index(drop=True), results_formatted], axis=1).to_string())

# %% [markdown]
# 9. Save the Model
# ----------------
# Let's save the best model and the scaler for future use.

# %%
# Create a directory for models if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Save the best model
model_path = '../models/best_model.joblib'
joblib.dump(best_model, model_path)

# Save the scaler
scaler_path = '../models/scaler.joblib'
joblib.dump(scaler, scaler_path)

print(f"\nBest model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")

# %% [markdown]
# 10. Conclusion
# --------------
# In this demo, we've built a machine learning pipeline to predict housing prices in California.
# We've explored the data, trained multiple models, and evaluated their performance.
# The best performing model can be used to make predictions on new data.

# %%
print("\nDemo completed successfully!")

# %% [markdown]
# To convert this script to a Jupyter notebook, run:
# ```
# jupyter nbconvert --to notebook --execute housing_price_prediction.py
# ```
# 
# This will create a new file called `housing_price_prediction.ipynb` that you can open and run in Jupyter.
