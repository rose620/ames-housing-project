cimport pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Suppress scikit-learn preprocessing category warnings from cluttering log
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing')

# Import custom modules
from src.pipeline import create_production_pipeline
from src.evaluation import run_model_audit

def main():
    # 1. Load dataset 
    data_path = os.path.join('data', 'train.csv')
    df = pd.read_csv(data_path)

    print("Loading and preparing Ames Housing dataset...")

    # Remove extreme outliers due to anomaly in Large Ground living area vs Sale Price
    df = df[df['GrLivArea'] < 4000] 
    # Remove clay tile feature also due to anomaly where one house shows discrepancy leading to class imbalance
    df = df[df['RoofMatl'] != 'ClyTile']
    
    # 2. Isolate and Define Target and Features
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # 3. Separate data pools
    numeric_cols = [col for col in X.select_dtypes(include=['int64', 'float64']).columns]
    categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

    # 4. Establish Train/Test Isolation Splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
        
    # 5. Initialize & Evaluate Baseline Lasso Pipeline
    print("\n--- Running Baseline LassoCV Model ---")
    lasso_pipeline = create_production_pipeline(numeric_cols, categorical_cols, model_type='lasso')
    lasso_pipeline.fit(X_train, y_train)
    run_model_audit(lasso_pipeline, X_train, X_test, y_train, y_test)
    
    # 6. Initialize & Evaluate Tuned XGBoost Pipeline
    print("\n--- Running Hyperparameter-Tuned XGBoost Model ---")
    xgb_pipeline = create_production_pipeline(numeric_cols, categorical_cols, model_type='xgb', tune_hyperparams=True)
    xgb_pipeline.fit(X_train, y_train)
    run_model_audit(xgb_pipeline, X_train, X_test, y_train, y_test)
    
    # 7. Evaluate the Ensemble Blend (70% Lasso / 30% XGBoost)
    print("\n--- Running Ensemble Blend (70% Lasso / 30% XGBoost) ---")
    
    # Generate predictions in clean dollar formats
    lasso_train_preds = lasso_pipeline.predict(X_train)
    lasso_test_preds = lasso_pipeline.predict(X_test)
    
    xgb_train_preds = xgb_pipeline.predict(X_train)
    xgb_test_preds = xgb_pipeline.predict(X_test)
    
    # Calculate the weighted blend
    ensemble_train_preds = (0.70 * lasso_train_preds) + (0.30 * xgb_train_preds)
    ensemble_test_preds = (0.70 * lasso_test_preds) + (0.30 * xgb_test_preds)
    
    # Calculate Metrics
    ensemble_r2_train = r2_score(y_train, ensemble_train_preds)
    ensemble_r2_test = r2_score(y_test, ensemble_test_preds)
    
    print("=========================================")
    print("       ENSEMBLE PERFORMANCE AUDIT        ")
    print("=========================================")
    print(f"Ensemble Training R² Score: {ensemble_r2_train:.4f}")
    print(f"Ensemble Testing R² Score:  {ensemble_r2_test:.4f}")
    print(f"Ensemble Generalization Delta: {abs(ensemble_r2_train - ensemble_r2_test):.4f}")
    print("=========================================")

if __name__ == "__main__":
    main()
