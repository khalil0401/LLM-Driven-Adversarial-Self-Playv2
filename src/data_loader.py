import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TONIoTLoader:
    """
    Data Loader for TON_IoT Dataset.
    Focuses on IoT telemetry and Network Flow features.
    """
    def __init__(self, filepath=None, window_size=10):
        self.window_size = window_size
        self.filepath = filepath
        self.scaler = MinMaxScaler()
        self.data = None
        self.labels = None
        self.feature_columns = [
            'src_port', 'dst_port', 'proto', 'conn_state', 
            'duration', 'src_bytes', 'dst_bytes'
        ] # Simplified subset for now
        
        self.load_data()
        
    def load_data(self):
        if self.filepath and os.path.exists(self.filepath):
            print(f"Loading TON_IoT from {self.filepath}...")
            df = pd.read_csv(self.filepath)
            # Basic preprocessing would go here
            # For now, we mock it if file doesn't exist for reproducibility logic
        else:
            print("Generating MOCK TON_IoT Data (Generic features)...")
            # Generate synthetic data mimicking IoT traffic
            # 3 Nodes (Devices), 1000 steps
            # State: [Load, Temp, Network_Out]
            n_samples = 2000
            
            # Normal behavior (Sine waves + noise)
            t = np.linspace(0, 100, n_samples)
            norm_traffic = np.sin(t) * 0.3 + 0.5 + np.random.normal(0, 0.05, n_samples)
            
            self.data = np.stack([
                norm_traffic, # Node 1
                np.roll(norm_traffic, 100), # Node 2
                np.roll(norm_traffic, 200)  # Node 3
            ], axis=1) # Shape (2000, 3)
            
            self.labels = np.zeros(n_samples) # 0 = Normal
            
            # Inject some anomalies for "Attack" labels in ground truth
            # t=500-600
            self.data[500:600, 0] += 0.8 # Spike
            self.labels[500:600] = 1 # Attack
            
            # Normalize to [0, 1]
            self.data = np.clip(self.data, 0, 1)

    def get_episode(self, episode_idx, length=200):
        """Returns a slice of data for an episode."""
        start = (episode_idx * length) % (len(self.data) - length)
        end = start + length
        return self.data[start:end], self.labels[start:end]
