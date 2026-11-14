import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


class SmearingLogTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps a regressor to train on log1p(y) and back-transform predictions
    to the original scale using Duan's smearing estimator to avoid retransformation bias.
    """
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        y_log = np.log1p(y)
        self.regressor_.fit(X, y_log)
        
        # Compute Duan's smearing factor from log-scale training residuals
        y_log_pred = self.regressor_.predict(X)
        residuals = y_log - y_log_pred
        self.smearing_factor_ = np.mean(np.exp(residuals))
        return self

    def predict(self, X):
        y_log_pred = self.regressor_.predict(X)
        # Apply exact smearing correction for log1p/expm1 transformation
        return (np.exp(y_log_pred) * self.smearing_factor_) - 1


def create_production_pipeline(numeric_cols, categorical_cols, model_type='lasso', tune_hyperparams=False):
    """
    Create a non-leaking preprocessing transformer coupled with either 
    LassoCV or XGBoost, incorporating strategic imputation and Duan's smearing 
    for automated target log-transformation.
    """
    # 1. Handle Strategic Imputation Categories
    # Define features where NaN means "Does Not Exist"(otherwise the large quantity of NaNs for PoolQC will seem like the majority of homes have a pool)
    structural_none_cols = [
        'PoolQC', 'Alley', 'Fence', 'MiscFeature', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
    ]
    
    # Isolate columns present in dataset's categorical list
    cat_none_cols = [col for col in structural_none_cols if col in categorical_cols]
    cat_freq_cols = [col for col in categorical_cols if col not in structural_none_cols]

    # 2. Configure Preprocessing Steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Path A: Missing values become a string constant ('None')
    cat_none_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # Path B: Missing values utilize most frequent strategy
    cat_freq_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle the preprocessing blocks together
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat_none', cat_none_transformer, cat_none_cols),
            ('cat_freq', cat_freq_transformer, cat_freq_cols)
        ]
    )

    # 3. Select and Configure the Base Regressor Component
    if model_type == 'xgb':
        regressor = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        regressor = LassoCV(cv=5, random_state=42, max_iter=50000)

    # 4. Wrap the Regressor to automate log transformations + Duan's smearing factor
    # added log due to rt skewed data on Lasso and noticed overfitting w/XGBoost
    # Trains on log1p(y) and outputs bias-corrected dollar predictions on .predict()
    wrapped_regressor = SmearingLogTargetRegressor(regressor=regressor)

    # 5. Bind Preprocessor and the Wrapped Model into Final Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', wrapped_regressor)
    ])
    
    # 6. Handle Hyperparameter Tuning for XGBoost only if requested for efficiency
    if model_type == 'xgb' and tune_hyperparams:
        # NOTE: Because XGBoost is nested inside SmearingLogTargetRegressor, 
        # which is nested inside the Pipeline, the naming keys use double regressor prefix -> 'regressor__regressor__'.
        param_grid = {
            'regressor__regressor__n_estimators': [100, 200, 300],
            'regressor__regressor__max_depth': [3, 4, 5],
            'regressor__regressor__learning_rate': [0.05, 0.1, 0.15],
            'regressor__regressor__subsample': [0.8, 1.0]
        }

        return GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )

    return full_pipeline
