import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from itertools import islice


ALPHA, LEARNING_RATE = 0.1, 0.1
K_STEPS, N_COUNTIES = 5, 100

def load_and_preprocess_data():
    df = (pd.read_csv('covid_data.csv')
          .assign(date=lambda x: pd.to_datetime(x['date'])))
    
    print("Available columns:", df.columns.tolist())
    
    county_col = next(iter(['county', 'fips']), None) if 'county' in df.columns else \
                 next((col for col in df if df[col].nunique() == N_COUNTIES 
                      and not pd.api.types.is_datetime64_any_dtype(df[col])), None)
    print(f"Selected county column: {county_col}")
    
    return df.assign(county_id=df[county_col])

def create_prediction_features(df, predict_days_ahead=14):
    return (df.sort_values('date')
            .groupby('county_id', group_keys=False)
            .apply(lambda g: g.assign(target=g.response.shift(-predict_days_ahead)))
            .dropna(subset=['target']))

def train_prediction_model(train_data):
    features = [c for c in train_data if c not in {'response', 'target', 'date', 'county_id'}]
    X, y = train_data[features], train_data.target
    
    scaler = StandardScaler().fit(X)
    model = RandomForestRegressor(n_estimators=200, random_state=42).fit(scaler.transform(X), y)
    
    print("Top features:")
    for feat, imp in sorted(zip(features, model.feature_importances_), 
                           key=lambda x: -x[1])[:10]:
        print(f"{feat}: {imp:.4f}")
    
    return model, scaler, features

def fit_weight_matrices(errors_history, k=K_STEPS):
    errors = np.array(errors_history)
    W = np.zeros((k, N_COUNTIES, N_COUNTIES))
    
    for target in range(N_COUNTIES):
        X = np.hstack([errors[k-i-1:-i-1] for i in range(k)])
        y = errors[k:, target]
        
        XtX = X.T @ X + 2 * np.eye(X.shape[1])
        XtX.flat[::X.shape[1]+1] += 1.0  # Add diagonal penalty
        weights = np.linalg.solve(XtX, X.T @ y)
        
        for i in range(k):
            W[i, target] = weights[i*N_COUNTIES:(i+1)*N_COUNTIES]
    
    return W

def adaptive_conformal_inference(y_pred, y_true, tau, eta=LEARNING_RATE, alpha=ALPHA):
    bounds = np.column_stack([y_pred - tau, y_pred + tau])
    covered = (y_true >= bounds[:, 0]) & (y_true <= bounds[:, 1])
    tau_new = np.fmax(tau - eta * (covered - (1 - alpha)), 0.01)
    return bounds, covered, tau_new

def quantile_tracking_prediction(y_pred, y_true, s_pred, tau, eta=LEARNING_RATE, alpha=ALPHA):
    log_pred = np.log1p(y_pred)
    bounds = np.expm1(np.column_stack([log_pred - (s_pred + tau), 
                                      log_pred + (s_pred + tau)]))
    bounds[:, 0] = np.fmax(bounds[:, 0], 0)
    covered = (y_true >= bounds[:, 0]) & (y_true <= bounds[:, 1])
    tau_new = np.fmax(tau - eta * (covered - (1 - alpha)), 0.01)
    return bounds, covered, tau_new

def evaluate_predictions(df, model, scaler, features, start, end):
    results = []
    counties = df.county_id.unique()
    tau_aci = np.full(N_COUNTIES, 15.0)
    tau_qt = np.full(N_COUNTIES, 0.5)
    errors = []
    

    warm_dates = (df.date[df.date < start - pd.Timedelta(days=14)]
                  .nlargest(30).tolist())
    
    for date in warm_dates:
        pred_date = date - pd.Timedelta(days=14)
        X = (df[df.date == pred_date]
             .groupby('county_id')[features]
             .first()
             .reindex(counties, fill_value=0))
        y_pred = model.predict(scaler.transform(X))
        y_true = (df[df.date == date]
                  .set_index('county_id').response
                  .reindex(counties, fill_value=0).values)
        
        errors.append(np.abs(np.log1p(y_true) - np.log1p(y_pred)))
    
    W = fit_weight_matrices(errors) if len(errors) >= K_STEPS else None
    
    for date in pd.date_range(start, end):
        pred_date = date - pd.Timedelta(days=14)
        X = (df[df.date <= pred_date]
             .groupby('county_id').last()[features]
             .reindex(counties, fill_value=0))
        y_pred = model.predict(scaler.transform(X))
        y_true = (df[df.date == date]
                  .set_index('county_id').response
                  .reindex(counties, fill_value=0).values)
        
        s_t = np.abs(np.log1p(y_true) - np.log1p(y_pred))
        s_pred = (sum(W[k] @ errors[-k-1] for k in range(K_STEPS)) 
                 if W is not None else np.mean([e.mean() for e in errors]))
        
        _, cov_aci, tau_aci = adaptive_conformal_inference(y_pred, y_true, tau_aci)
        _, cov_qt, tau_qt = quantile_tracking_prediction(y_pred, y_true, s_pred, tau_qt)
        
        errors.append(s_t)
        if len(errors) > K_STEPS + 30: errors.pop(0)
        
        results.append({
            'date': date,
            'aci_coverage': cov_aci.mean(),
            'qt_coverage': cov_qt.mean(),
            'avg_pred': y_pred.mean(),
            'avg_true': y_true.mean(),
            **dict(zip(['tau_aci_avg', 'tau_qt_avg'], 
                      [tau_aci.mean(), tau_qt.mean()]))
        })
    
    return pd.DataFrame(results)
