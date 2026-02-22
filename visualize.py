import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_with_anomalies(df: pd.DataFrame, metrics: list, anomaly_col: str, title: str, save_path: str = None):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, metrics):
        ax.plot(df['timestamp'], df[metric], label=metric, color='blue', alpha=0.7)
        
        anomalies = df[df[anomaly_col] == 1]
        ax.scatter(anomalies['timestamp'], anomalies[metric], color='red', label='Anomaly', zorder=5)
        
        ax.set_ylabel(metric)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Timestamp')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import os
    if os.path.exists("analyzed_metrics.csv"):
        data = pd.read_csv("analyzed_metrics.csv")
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        plot_metrics_with_anomalies(
            data, 
            ['job_runtime', 'memory_usage', 'throughput'], 
            'global_anomaly_iforest', 
            'Isolation Forest Detected Anomalies',
            save_path='iforest_anomalies.png'
        )
    else:
        print("analyzed_metrics.csv not found.")
