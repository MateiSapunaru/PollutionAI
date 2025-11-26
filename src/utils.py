import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def get_data_info(df):
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numerical_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist()
    }
    return info


def encode_categorical_features(df, categorical_cols, method='label'):
    df_encoded = df.copy()
    encoders = {}
    
    if method == 'label':
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    return df_encoded, encoders


def scale_features(X_train, X_test, numerical_cols):
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)


def calculate_regression_metrics(y_true, y_pred):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2_Score': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics


def calculate_classification_metrics(y_true, y_pred, labels=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1_Score_Weighted': f1_score(y_true, y_pred, average='weighted'),
        'F1_Score_Macro': f1_score(y_true, y_pred, average='macro'),
        'Classification_Report': classification_report(y_true, y_pred, target_names=labels),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', figsize=(10, 8), save_path=None):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_importance(importance_df, top_n=20, title='Feature Importance', figsize=(12, 8), save_path=None):
    plt.figure(figsize=figsize)
    
    # Sort and get top N features
    top_features = importance_df.nlargest(top_n, 'importance')
    
    # Create bar plot
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted', figsize=(10, 6), save_path=None):
    plt.figure(figsize=figsize)
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_residuals(y_true, y_pred, title='Residual Plot', figsize=(10, 6), save_path=None):
    residuals = y_true - y_pred
    
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, alpha=0.5, s=30)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def print_metrics(metrics, title="Model Performance Metrics"):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    
    for key, value in metrics.items():
        if key not in ['Classification_Report', 'Confusion_Matrix']:
            if isinstance(value, float):
                print(f"{key:.<40} {value:.4f}")
            else:
                print(f"{key:.<40} {value}")
    
    print("="*60 + "\n")


def save_predictions(y_true, y_pred, filename='predictions.csv'):
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': y_true - y_pred
    })
    
    results_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


if __name__ == "__main__":
    print("Utility Functions Module")
    print("This module contains shared functions for the Soil Pollution & Disease Detection Project")
