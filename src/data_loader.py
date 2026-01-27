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
            
            # Select relevant columns
            cols = ['src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes', 'proto', 'conn_state', 'label']
            
            # Handle missing columns if partial dataset
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols].copy()
            
            # Preprocessing: Categorical Encoding
            if 'proto' in df.columns:
                df['proto'] = pd.Categorical(df['proto']).codes
            if 'conn_state' in df.columns:
                df['conn_state'] = pd.Categorical(df['conn_state']).codes
                
            # Fill NaNs
            df = df.fillna(0)
            
            # Store labels
            if 'label' in df.columns:
                self.labels = df['label'].values
                feature_df = df.drop(columns=['label'])
            else:
                self.labels = np.zeros(len(df))
                feature_df = df
            
            # Normalize Features
            self.data = self.scaler.fit_transform(feature_df)
            
            # Pad to 3 dimensions if fewer features (Env expects 3 Nodes/Features roughly)
            # Actually Env DataDrivenCPSEnv expects whatever data is returned.
            # But obs space is Box(6). 
            # We need to map these >6 features to 3 "Nodes" concept or PCA them.
            # For simplicity, we select top 3 logical features for the "Node Values":
            # Node 1 ~ Traffic Volume (src_bytes)
            # Node 2 ~ Duration (duration)
            # Node 3 ~ Connection State (conn_state)
            
            # Helper to map to 3 dim for compatibility with DataDrivenCPSEnv default 3-node view
            # Or we update DataDrivenCPSEnv to handle correct dims. 
            # Let's map key features to the first 3 columns for visual intuition.
            
            # Target Features: src_bytes, dst_bytes, duration
            target_cols = ['src_bytes', 'dst_bytes', 'duration']
            indices = [feature_df.columns.get_loc(c) for c in target_cols if c in feature_df.columns]
            
            if len(indices) >= 3:
                self.data = self.data[:, indices[:3]] # Take top 3
            else:
                # Pad if not enough
                current_dim = self.data.shape[1]
                if current_dim < 3:
                    padding = np.zeros((len(self.data), 3 - current_dim))
                    self.data = np.hstack([self.data, padding])
                self.data = self.data[:, :3] # Force 3 dim
                
            print(f"Loaded {len(self.data)} samples. Mapped to 3 CPS Nodes.")
            
        else:
            print("Generating MOCK TON_IoT Data (Generic features)...")
            # ... (Mock logic remains)
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
