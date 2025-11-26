import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class DiseaseAI:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.disease_type_models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.best_disease_model_name = None
        self.best_disease_model = None
        self.disease_type_encoder = None

        self.output_dir = 'models'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}/")

        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def preprocess_data(self):
        print("\n" + "="*80)
        print("DATA PREPROCESSING - DISEASE TYPE".center(80))
        print("="*80)

        feature_cols = [
            'Pollutant_Type',
            'Total_Concentration_mg_kg',
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
            'Water_Source_Type',
            'Health_Symptoms',
            'Age_Group_Affected',
            'Gender_Most_Affected'
        ]

        target_col = 'Disease_Type'

        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()

        print(f"\nFeatures selected: {len(feature_cols)}")
        print(f"Target variable: {target_col}")

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        print(f"\nCategorical features: {len(categorical_cols)}")
        print(f"Numerical features: {len(numerical_cols)}")

        print("\nEncoding categorical variables...")
        for col in categorical_cols:
            if col not in self.encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))

        print("Encoding completed")

        self.disease_type_encoder = LabelEncoder()
        y_encoded = self.disease_type_encoder.fit_transform(y)
        target_classes = self.disease_type_encoder.classes_

        print(f"\nTarget classes: {list(target_classes)}")
        print(f"Number of classes: {len(target_classes)}")

        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")

        print("\nScaling features...")
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        print("Scaling completed")
        print("="*80 + "\n")

        return X_train_scaled, X_test_scaled, y_train, y_test, target_classes

    def train_disease_type_models(self, X_train, X_test, y_train, y_test, target_classes):
        print("\n" + "="*80)
        print("DISEASE TYPE MODEL TRAINING".center(80))
        print("="*80)

        # Define models
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        }

        results = {}

        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')

            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'y_pred_test': y_pred_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'classification_report': classification_report(
                    y_test, y_pred_test, target_names=target_classes
                )
            }

            self.disease_type_models[name] = model

            print(f"   Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
            print(f"   Train F1:       {train_f1:.4f} | Test F1:       {test_f1:.4f}")
            print(f"   Test Precision: {test_precision:.4f}")
            print(f"   Test Recall:    {test_recall:.4f}")

        print("\n" + "="*80 + "\n")

        return results, y_test, target_classes

    def compare_models(self, results, model_type='disease_type'):
        print("\n" + "="*80)
        print(f"MODEL COMPARISON - {model_type.upper()}".center(80))
        print("="*80 + "\n")

        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Test_Accuracy': result['test_accuracy'],
                'Test_F1': result['test_f1'],
                'Test_Precision': result['test_precision'],
                'Test_Recall': result['test_recall']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_F1', ascending=False)

        print(comparison_df.to_string(index=False))
        print("\n" + "="*80)

        best_model_name = comparison_df.iloc[0]['Model']
        self.best_disease_model_name = best_model_name
        self.best_disease_model = results[best_model_name]['model']

        print(f"\nBest Disease Type Model: {best_model_name}")
        print(f"   Test Accuracy: {comparison_df.iloc[0]['Test_Accuracy']:.4f}")
        print(f"   Test F1 Score: {comparison_df.iloc[0]['Test_F1']:.4f}")
        print(f"   Test Precision: {comparison_df.iloc[0]['Test_Precision']:.4f}")
        print(f"   Test Recall: {comparison_df.iloc[0]['Test_Recall']:.4f}")

        print("\n" + "="*80 + "\n")

        return comparison_df

    def plot_confusion_matrices(self, results, target_classes, model_type='disease_type', save=True):

        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.ravel()

        for idx, (name, result) in enumerate(results.items()):
            cm = result['confusion_matrix']

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_classes, yticklabels=target_classes,
                ax=axes[idx], cbar_kws={"shrink": 0.8}
            )
            axes[idx].set_title(
                f'{name}\nAccuracy: {result["test_accuracy"]:.4f}',
                fontweight='bold', fontsize=12
            )
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)

        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(self.output_dir, f'{model_type}_confusion_matrices.png'),
                dpi=300, bbox_inches='tight'
            )
        plt.show()

    def plot_model_comparison(self, results, model_type='disease_type', save=True):
        model_names = list(results.keys())
        test_acc = [results[name]['test_accuracy'] for name in model_names]
        test_f1 = [results[name]['test_f1'] for name in model_names]
        test_precision = [results[name]['test_precision'] for name in model_names]
        test_recall = [results[name]['test_recall'] for name in model_names]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].barh(model_names, test_acc, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Model Comparison - Accuracy', fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlim([0, 1])

        axes[0, 1].barh(model_names, test_f1, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_title('Model Comparison - F1 Score', fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlim([0, 1])

        axes[1, 0].barh(model_names, test_precision, color='seagreen', edgecolor='black')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_title('Model Comparison - Precision', fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlim([0, 1])

        axes[1, 1].barh(model_names, test_recall, color='gold', edgecolor='black')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_title('Model Comparison - Recall', fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlim([0, 1])

        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(self.output_dir, f'{model_type}_model_comparison.png'),
                dpi=300, bbox_inches='tight'
            )
        plt.show()

    def save_model(self, filepath='disease_type_model.pkl'):

        if not os.path.dirname(filepath):
            filepath = os.path.join(self.output_dir, filepath)

        disease_package = {
            'model': self.best_disease_model,
            'model_name': self.best_disease_model_name,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_encoder': self.disease_type_encoder
        }
        joblib.dump(disease_package, filepath)
        print(f"Disease type model saved to {filepath}")

    def predict_disease_type(self, input_data):

        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        for col, encoder in self.encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))

        input_scaled = self.scaler.transform(input_df[self.feature_names])

        disease_pred = self.best_disease_model.predict(input_scaled)
        disease_type = self.disease_type_encoder.inverse_transform(disease_pred)[0]

        return disease_type


def main():
    """
    Main function to train and evaluate the Disease Type AI model
    """
    print("\n" + "="*80)
    print("DISEASE AI - DISEASE TYPE PREDICTION".center(80))
    print("="*80 + "\n")

    filepath = 'data/soil_contamination_scientific.csv'
    model = DiseaseAI(filepath)

    print("\n" + "▶"*40)
    print("PART 1: DISEASE TYPE PREDICTION")
    print("▶"*40 + "\n")

    X_train, X_test, y_train, y_test, target_classes = model.preprocess_data()
    disease_results, y_test_disease, disease_classes = model.train_disease_type_models(
        X_train, X_test, y_train, y_test, target_classes
    )
    disease_comparison = model.compare_models(disease_results, model_type='disease_type')
    model.plot_confusion_matrices(
        disease_results, disease_classes, model_type='disease_type', save=True
    )
    model.plot_model_comparison(
        disease_results, model_type='disease_type', save=True
    )

    model.save_model('disease_type_model.pkl')

    print("\n" + "="*80)
    print("DISEASE TYPE AI MODEL TRAINING COMPLETED!".center(80))
    print(f"All outputs saved to '{model.output_dir}/' folder".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
