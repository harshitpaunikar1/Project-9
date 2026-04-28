"""
Linear regression with cross-validation analysis.
Demonstrates k-fold CV, LOOCV, regularization comparison, and learning curves.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import (
        KFold, LeaveOneOut, cross_val_score, learning_curve, train_test_split
    )
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CrossValidatedLinearModel:
    """
    Trains and cross-validates linear regression and regularized variants.
    Reports RMSE, MAE, R2 across k-fold splits and generates learning curve data.
    """

    def __init__(self, feature_cols: List[str], target_col: str,
                 cv_folds: int = 5, random_state: int = 42):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None

    def _models(self) -> Dict:
        return {
            "OLS": LinearRegression(),
            "Ridge(a=0.1)": Ridge(alpha=0.1),
            "Ridge(a=1)": Ridge(alpha=1.0),
            "Ridge(a=10)": Ridge(alpha=10.0),
            "Lasso(a=0.01)": Lasso(alpha=0.01, max_iter=3000),
            "Lasso(a=0.1)": Lasso(alpha=0.1, max_iter=3000),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000),
        }

    def fit_and_compare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run k-fold CV for all models. Returns comparison DataFrame sorted by CV RMSE."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        X = df[self.feature_cols].fillna(df[self.feature_cols].median()).values
        y = df[self.target_col].values
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scaler = StandardScaler()
        self.results = []

        for name, model in self._models().items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            cv_neg_rmse = cross_val_score(pipe, X, y, cv=kf,
                                          scoring="neg_root_mean_squared_error")
            cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
            self.results.append({
                "model": name,
                "cv_rmse_mean": round(float(-cv_neg_rmse.mean()), 4),
                "cv_rmse_std": round(float(cv_neg_rmse.std()), 4),
                "cv_r2_mean": round(float(cv_r2.mean()), 4),
                "cv_r2_std": round(float(cv_r2.std()), 4),
            })

        results_df = pd.DataFrame(self.results).sort_values("cv_rmse_mean").reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def learning_curve_data(self, df: pd.DataFrame,
                             model_name: str = "Ridge(a=1)") -> pd.DataFrame:
        """Compute training and validation scores for increasing training set sizes."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        model = self._models().get(model_name, Ridge(alpha=1.0))
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        X = df[self.feature_cols].fillna(df[self.feature_cols].median()).values
        y = df[self.target_col].values
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X, y, cv=self.cv_folds,
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=self.random_state,
        )
        return pd.DataFrame({
            "train_size": train_sizes,
            "train_rmse": (-train_scores.mean(axis=1)).round(4),
            "val_rmse": (-val_scores.mean(axis=1)).round(4),
            "train_rmse_std": train_scores.std(axis=1).round(4),
            "val_rmse_std": val_scores.std(axis=1).round(4),
        })

    def loocv_score(self, df: pd.DataFrame) -> Dict:
        """Run Leave-One-Out CV on OLS for small datasets."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        X = df[self.feature_cols].fillna(df[self.feature_cols].median()).values
        y = df[self.target_col].values
        loo = LeaveOneOut()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        scores = cross_val_score(pipe, X, y, cv=loo, scoring="neg_root_mean_squared_error")
        return {
            "loocv_rmse": round(float(-scores.mean()), 4),
            "n_splits": len(scores),
        }

    def polynomial_feature_analysis(self, df: pd.DataFrame,
                                     max_degree: int = 4) -> pd.DataFrame:
        """Compare Ridge regression with polynomial feature expansion up to max_degree."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        X = df[self.feature_cols].fillna(df[self.feature_cols].median()).values
        y = df[self.target_col].values
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        records = []
        for degree in range(1, max_degree + 1):
            pipe = Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ])
            cv_scores = cross_val_score(pipe, X, y, cv=kf,
                                        scoring="neg_root_mean_squared_error")
            records.append({
                "degree": degree,
                "cv_rmse": round(float(-cv_scores.mean()), 4),
                "cv_rmse_std": round(float(cv_scores.std()), 4),
            })
        return pd.DataFrame(records)

    def assumption_checks(self, df: pd.DataFrame) -> Dict:
        """Check basic regression assumptions on the training residuals."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        X = df[self.feature_cols].fillna(df[self.feature_cols].median()).values
        y = df[self.target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_s, y_train)
        residuals = y_train - model.predict(X_train_s)
        try:
            from scipy.stats import shapiro
            _, p_normality = shapiro(residuals[:500] if len(residuals) > 500 else residuals)
            normality_p = round(float(p_normality), 4)
        except ImportError:
            normality_p = None
        return {
            "residual_mean": round(float(residuals.mean()), 6),
            "residual_std": round(float(residuals.std()), 4),
            "residual_skew": round(float(pd.Series(residuals).skew()), 4),
            "normality_p_value": normality_p,
            "r2_train": round(float(r2_score(y_train, model.predict(X_train_s))), 4),
            "r2_test": round(float(r2_score(y_test, model.predict(X_test_s))), 4),
        }


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    noise = np.random.normal(0, 0.5, n)
    y = 3.5 * X1 - 2.0 * X2 + 1.5 * X3 + noise

    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "y": y})
    cv_model = CrossValidatedLinearModel(
        feature_cols=["X1", "X2", "X3"],
        target_col="y",
        cv_folds=5,
    )

    print("Cross-validation comparison:")
    results = cv_model.fit_and_compare(df)
    print(results.to_string(index=False))
    print(f"\nBest model: {cv_model.best_model_name}")

    lc = cv_model.learning_curve_data(df, model_name="Ridge(a=1)")
    print("\nLearning curve (validation RMSE vs training size):")
    print(lc[["train_size", "train_rmse", "val_rmse"]].to_string(index=False))

    poly = cv_model.polynomial_feature_analysis(df, max_degree=3)
    print("\nPolynomial degree analysis:")
    print(poly.to_string(index=False))

    checks = cv_model.assumption_checks(df)
    print("\nAssumption checks:", checks)

    loocv = cv_model.loocv_score(df.head(100))
    print("LOOCV RMSE:", loocv)
