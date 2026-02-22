import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class BaselineDetector:
    def __init__(self, window: int = 50, z_threshold: float = 3.0):
        self.window = window
        self.z_threshold = z_threshold
        
    def detect(self, series: pd.Series) -> pd.Series:
        assert series.notna().all(), "Input series contains NaNs."
        
        rolling = series.rolling(window=self.window, min_periods=1)
        mean = rolling.mean()
        std = rolling.std().bfill().fillna(0)
        
        std = np.where(std == 0, 1e-6, std)
        z_scores = (series - mean) / std
        anomalies = np.abs(z_scores) > self.z_threshold
        return anomalies.astype(int)

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        
    def detect(self, df: pd.DataFrame, features: list) -> pd.Series:
        X = df[features].values
        assert not np.isnan(X).any(), "Input features contain NaNs."
        
        preds = self.model.fit_predict(X)
        anomalies = (preds == -1).astype(int)
        return pd.Series(anomalies, index=df.index)

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    results = df.copy()
    
    baseline_detector = BaselineDetector(window=50, z_threshold=3.0)
    for col in ['job_runtime', 'memory_usage', 'throughput']:
        results[f'{col}_anomaly_zscore'] = baseline_detector.detect(results[col])
        
    if_detector = IsolationForestDetector(contamination=0.05)
    features = ['job_runtime', 'memory_usage', 'throughput']
    results['global_anomaly_iforest'] = if_detector.detect(results, features)
    
    return results

if __name__ == "__main__":
    import os
    if os.path.exists("synthetic_metrics.csv"):
        data = pd.read_csv("synthetic_metrics.csv")
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        analyzed = run_anomaly_detection(data)
        analyzed.to_csv("analyzed_metrics.csv", index=False)
        print("Anomaly detection complete. Saved to analyzed_metrics.csv.")
    else:
        print("synthetic_metrics.csv not found.")
