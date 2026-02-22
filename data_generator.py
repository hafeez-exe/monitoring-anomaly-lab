import numpy as np
import pandas as pd

def generate_performance_metrics(
    n_samples: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic time-series performance metrics with noise, drift, and regressions.
    """
    np.random.seed(seed)
    time = np.arange(n_samples)
    
    runtime = 120.0 + np.random.normal(0, 5.0, n_samples)
    memory = 4096.0 + np.random.normal(0, 50.0, n_samples)
    throughput = 500.0 + np.random.normal(0, 20.0, n_samples)
    
    drift_start = int(n_samples * 0.3)
    memory[drift_start:] += np.linspace(0, 1000, n_samples - drift_start)
    
    reg1_idx = int(n_samples * 0.6)
    runtime[reg1_idx : reg1_idx + 50] += 50.0
    
    reg2_idx = int(n_samples * 0.8)
    throughput[reg2_idx : reg2_idx + 20] -= 200.0
    
    mem_spike_idx = int(n_samples * 0.45)
    memory[mem_spike_idx : mem_spike_idx + 5] += 2000.0

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=n_samples, freq='h'),
        'job_runtime': runtime,
        'memory_usage': memory,
        'throughput': throughput
    })
    
    assert not df.isna().any().any(), "Generated data contains NaNs."
    return df

if __name__ == "__main__":
    df = generate_performance_metrics()
    df.to_csv("synthetic_metrics.csv", index=False)
    print(f"Generated synthetic_metrics.csv with {len(df)} samples.")
