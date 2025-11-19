"""
Concentration AI Model - Pollutant Concentration Prediction
Predicts pollutant concentration levels based on environmental and soil factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ConcentrationAI:
    """
    AI Model for predicting pollutant concentration levels
    """
    
    def __init__(self, filepath):
        """
        Initialize the model with dataset
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        """
        self.df = pd.read_csv(filepath)
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.best_model_name = None
        self.best_model = None
        
        # Create output directory for model artifacts
        self.output_dir = 'models'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created output directory: {self.output_dir}/")
        
        print(f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def preprocess_data(self):
        """
        Preprocess data for model training
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : Split datasets
        """
        print("\n" + "="*80)
        print("DATA PREPROCESSING".center(80))
        print("="*80)
        
        # Select features
        feature_cols = [
            'Pollutant_Type',
            'Bioavailable_Concentration_mg_kg',
            'Soil_pH',
            'Soil_Texture',
            'Soil_Organic_Matter_%',
            'CEC_meq_100g',
            'Temperature_C',
            'Humidity_%',
            'Rainfall_mm',
            'Region',
            'Country',
            'Crop_Type',
            'Farming_Practice',
            'Nearby_Industry',
            'Distance_from_Source_km',
            'Years_Since_Contamination',
            'Water_Source_Type'
        ]
        
        target_col = 'Total_Concentration_mg_kg'
        
        # Create feature and target datasets
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        print(f"\n✓ Features selected: {len(feature_cols)}")
        print(f"✓ Target variable: {target_col}")
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n📊 Categorical features: {len(categorical_cols)}")
        print(f"📊 Numerical features: {len(numerical_cols)}")
        
        # Encode categorical variables
        print("\n🔄 Encoding categorical variables...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        print("✓ Encoding completed")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"\n📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Testing set: {X_test.shape[0]} samples")
        
        # Scale features
        print("\n⚖️ Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print("✓ Scaling completed")
        print("="*80 + "\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple regression models and compare performance
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : Training and testing datasets
        
        Returns:
        --------
        dict : Dictionary of trained models and their performance
        """
        print("\n" + "="*80)
        print("MODEL TRAINING".center(80))
        print("="*80)
        
        # Define models
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\n🔧 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Store results
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'y_pred_test': y_pred_test
            }
            
            self.models[name] = model
            
            print(f"   ✓ Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
            print(f"   ✓ Train MAE:  {train_mae:.4f} | Test MAE:  {test_mae:.4f}")
            print(f"   ✓ Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")
        
        print("\n" + "="*80 + "\n")
        
        return results, y_test
    
    def compare_models(self, results):
        """
        Compare model performances
        
        Parameters:
        -----------
        results : dict
            Dictionary of model results
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON".center(80))
        print("="*80 + "\n")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Test_RMSE': result['test_rmse'],
                'Test_MAE': result['test_mae'],
                'Test_R2': result['test_r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        print(comparison_df.to_string(index=False))
        print("\n" + "="*80)
        
        # Select best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   ✓ Test R² Score: {comparison_df.iloc[0]['Test_R2']:.4f}")
        print(f"   ✓ Test RMSE: {comparison_df.iloc[0]['Test_RMSE']:.4f}")
        print(f"   ✓ Test MAE: {comparison_df.iloc[0]['Test_MAE']:.4f}")
        
        print("\n" + "="*80 + "\n")
        
        return comparison_df
    
    def plot_results(self, results, y_test, save=True):
        """
        Plot model performance results
        
        Parameters:
        -----------
        results : dict
            Dictionary of model results
        y_test : array-like
            True test values
        save : bool
            Whether to save plots
        """
        # Plot 1: Model Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(results.keys())
        test_rmse = [results[name]['test_rmse'] for name in model_names]
        test_mae = [results[name]['test_mae'] for name in model_names]
        test_r2 = [results[name]['test_r2'] for name in model_names]
        
        # RMSE Comparison
        axes[0, 0].barh(model_names, test_rmse, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('RMSE')
        axes[0, 0].set_title('Model Comparison - RMSE (Lower is Better)', fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # MAE Comparison
        axes[0, 1].barh(model_names, test_mae, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('MAE')
        axes[0, 1].set_title('Model Comparison - MAE (Lower is Better)', fontweight='bold')
        axes[0, 1].invert_yaxis()
        
        # R² Comparison
        axes[1, 0].barh(model_names, test_r2, color='seagreen', edgecolor='black')
        axes[1, 0].set_xlabel('R² Score')
        axes[1, 0].set_title('Model Comparison - R² Score (Higher is Better)', fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Actual vs Predicted for best model
        best_pred = results[self.best_model_name]['y_pred_test']
        axes[1, 1].scatter(y_test, best_pred, alpha=0.5, s=30)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Concentration (mg/kg)')
        axes[1, 1].set_ylabel('Predicted Concentration (mg/kg)')
        axes[1, 1].set_title(f'Actual vs Predicted - {self.best_model_name}', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'concentration_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Residual Analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        residuals = y_test.values - best_pred
        
        # Residual plot
        axes[0].scatter(best_pred, residuals, alpha=0.5, s=30)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residual Plot - {self.best_model_name}', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=50, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals', fontweight='bold')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'concentration_residual_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, save=True):
        """
        Analyze and plot feature importance
        
        Parameters:
        -----------
        save : bool
            Whether to save plots
        """
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS".center(80))
        print("="*80 + "\n")
        
        # Get feature importance from best model
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print("-" * 80)
            for idx, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:40s} : {row['importance']:.4f}")
            
            # Plot
            plt.figure(figsize=(12, 8))
            top_n = min(20, len(importance_df))
            sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'concentration_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\n" + "="*80 + "\n")
            
            return importance_df
        else:
            print("⚠️ Feature importance not available for this model type")
            return None
    
    def save_model(self, filepath='concentration_model.pkl'):
        """
        Save the trained model and preprocessors
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (default saves to models/ directory)
        """
        # If only filename provided, save to output directory
        if not os.path.dirname(filepath):
            filepath = os.path.join(self.output_dir, filepath)
        
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_package, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def predict_concentration(self, input_data):
        """
        Predict pollutant concentration for new data
        
        Parameters:
        -----------
        input_data : dict or pd.DataFrame
            Input features
            
        Returns:
        --------
        float : Predicted concentration
        """
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Encode categorical variables
        for col, encoder in self.encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Scale features
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Predict
        prediction = self.best_model.predict(input_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction


def main():
    """
    Main function to train and evaluate the Concentration AI model
    """
    print("\n" + "="*80)
    print("CONCENTRATION AI - POLLUTANT CONCENTRATION PREDICTION".center(80))
    print("="*80 + "\n")
    
    # Initialize model
    filepath = 'data/soil_contamination_scientific.csv'
    model = ConcentrationAI(filepath)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = model.preprocess_data()
    
    # Train models
    results, y_test = model.train_models(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df = model.compare_models(results)
    
    # Plot results
    model.plot_results(results, y_test, save=True)
    
    # Feature importance
    importance_df = model.feature_importance_analysis(save=True)
    
    # Save model
    model.save_model('concentration_model.pkl')
    
    print("\n" + "="*80)
    print("✓ CONCENTRATION AI MODEL TRAINING COMPLETED!".center(80))
    print(f"✓ All outputs saved to '{model.output_dir}/' folder".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()