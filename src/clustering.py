from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterator
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import logging
import time
import joblib
from dataclasses import dataclass
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryTracker:
    """Track GPU memory usage"""
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9
            }
        return {}

class ProgressTracker:
    """Track training progress and metrics"""
    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()
    
    def update(self, metrics: Dict[str, float]):
        self.metrics_history.append({
            'timestamp': time.time() - self.start_time,
            **metrics
        })
    
    def get_summary(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        return {
            'duration': time.time() - self.start_time,
            'final_metrics': self.metrics_history[-1]
        }

@dataclass
class ClusteringConfig:
    n_clusters: int = 16384  # Matching SAE latent dimension
    batch_size: int = 32
    n_init: int = 3
    max_iter: int = 100
    random_state: int = 42
    seq_len: int = 128
    n_features: int = 2304
    max_points_per_batch: int = 100_000
    
    def __post_init__(self):
        if self.n_clusters > self.max_points_per_batch:
            raise ValueError("max_points_per_batch must be larger than n_clusters")
        self.memory_estimate = (self.n_clusters * self.n_features * 4) / 1e9

class PositionalKMeans(nn.Module):
    def __init__(self, config: ClusteringConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_tracker = MemoryTracker()
        self.progress_tracker = ProgressTracker()
        
        logger.info(f"Initializing K-means with {config.n_clusters} clusters")
        logger.info(f"Estimated memory: {config.memory_estimate:.2f} GB")
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
            max_no_improvement=20,
            verbose=1
        )
        
        # Only store counts, load centroids when needed
        self.register_buffer('position_counts', 
            torch.zeros(config.seq_len, config.n_clusters, dtype=torch.uint32))
    
    def _batch_generator(self, chunk_paths: List[Path]) -> Iterator[torch.Tensor]:
        """Memory-efficient batch generator"""
        buffer = []
        buffer_size = 0
        
        for path in chunk_paths:
            try:
                chunk = np.load(path)['activations']
                chunk = chunk.reshape(-1, chunk.shape[-1])
                
                for idx in range(0, len(chunk), self.config.batch_size):
                    batch = torch.from_numpy(chunk[idx:idx + self.config.batch_size])
                    buffer.append(batch)
                    buffer_size += len(batch)
                    
                    if buffer_size >= self.config.max_points_per_batch:
                        yield torch.cat(buffer).to(self.device)
                        buffer = []
                        buffer_size = 0
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
                
        if buffer:
            yield torch.cat(buffer).to(self.device)
    
    def fit(self, chunk_paths: List[Path]):
        """Fit k-means with memory-efficient batching and progress tracking"""
        logger.info("Starting global k-means fitting...")
        torch.cuda.empty_cache()
        
        try:
            # Fit using generator
            for batch in tqdm(self._batch_generator(chunk_paths), desc="Fitting clusters"):
                self.kmeans.partial_fit(batch.cpu().numpy())
                
                # Track progress
                metrics = self.compute_metrics()
                self.progress_tracker.update(metrics)
                logger.info(f"Batch metrics: {metrics}")
                
            # Compute position statistics
            self._compute_position_statistics(chunk_paths)
            
        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            raise
        
        return self
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute clustering quality metrics"""
        metrics = {
            'inertia': self.kmeans.inertia_,
            'n_empty_clusters': (self.position_counts.sum(0) == 0).sum().item(),
        }
        
        # Add memory stats
        metrics.update(self.memory_tracker.get_memory_stats())
        
        # Position-specific metrics
        pos_entropy = torch.zeros(self.config.seq_len)
        for pos in range(self.config.seq_len):
            dist = self.get_position_distribution(pos)
            pos_entropy[pos] = -(dist * torch.log(dist + 1e-10)).sum()
        
        metrics.update({
            'position_entropy_mean': pos_entropy.mean().item(),
            'position_entropy_std': pos_entropy.std().item()
        })
        
        return metrics
    
    def get_position_distribution(self, position: int) -> torch.Tensor:
        """Get normalized cluster usage distribution for a position"""
        counts = self.position_counts[position].float()
        return counts / (counts.sum() + 1e-10)
    
    @classmethod
    def load_checkpoint(cls, path: Path) -> 'PositionalKMeans':
        """Load with validation"""
        try:
            state = torch.load(path / "clustering_results.pt")
            kmeans = joblib.load(path / "kmeans_model.joblib")
            
            instance = cls(state['config'])
            instance.kmeans = kmeans
            instance.position_counts = state['position_counts']
            
            # Validate loaded state
            metrics = instance.compute_metrics()
            logger.info(f"Loaded checkpoint metrics: {metrics}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def save_checkpoint(self, path: Path):
        """Save with metadata"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save torch components
        torch.save({
            'config': self.config,
            'position_counts': self.position_counts,
            'metrics': self.progress_tracker.get_summary()
        }, path / "clustering_results.pt")
        
        # Save sklearn model
        joblib.dump(self.kmeans, path / "kmeans_model.joblib")
        
        logger.info(f"Saved checkpoint to {path}")

def main():
    config = ClusteringConfig()
    input_dir = Path("/workspace/data/whitened_activations")
    output_dir = Path("/workspace/data/clustering_results")
    
    chunk_paths = sorted(input_dir.glob("whitened_chunk_*.npz"))
    logger.info(f"Found {len(chunk_paths)} whitened chunks")
    
    clustering = PositionalKMeans(config)
    clustering.fit(chunk_paths)
    
    # Save results
    clustering.save_checkpoint(output_dir)
    logger.info("Completed clustering")
    
    # Log final metrics
    final_metrics = clustering.progress_tracker.get_summary()
    logger.info(f"Final metrics: {final_metrics}")

if __name__ == "__main__":
    main()