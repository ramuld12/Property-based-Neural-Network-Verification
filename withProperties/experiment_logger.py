"""Experiment tracking and result logging for property-based training experiments.

This module provides utilities to automatically capture experiment configurations,
training metrics, and evaluation results for easy comparison and reproducibility.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class PropertyConfig:
    """Configuration for a single property constraint."""
    name: str
    feature_name: str
    feature_idx: int
    theta: float  # Original threshold
    theta_scaled: float  # Scaled threshold


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment run."""
    
    # Dataset configuration
    dataset_path: str
    num_features: int
    features: List[str]
    num_samples: int
    num_train_samples: int
    num_test_samples: int
    label_distribution: Dict[str, int]
    
    # Property configuration
    properties: List[PropertyConfig]
    lambda_prop: float  # Property loss weight
    
    # Training hyperparameters
    learning_rate: float
    batch_size: int
    num_epochs: int
    test_size: float
    
    # Optional metadata
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling nested dataclasses."""
        d = asdict(self)
        d["properties"] = [asdict(p) for p in self.properties]
        return d


@dataclass
class EpochMetrics:
    """Metrics recorded for a single epoch."""
    epoch: int
    pred_loss: float
    prop_loss: float
    total_loss: float
    prop_stats: Dict[str, float] = field(default_factory=dict)


class ExperimentLogger:
    """Tracks training and evaluation metrics for an experiment.
    
    Automatically organizes results by timestamp and number of properties
    in: experiments/{TIMESTAMP}_{NUMPROPS}props/config_and_results.json
    """
    
    def __init__(self, config: ExperimentConfig, base_dir: str = "experiments"):
        """Initialize experiment logger.
        
        Args:
            config: ExperimentConfig with all experiment settings
            base_dir: Root directory for experiments (default: "experiments")
        """
        self.config = config
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_props = len(config.properties)
        self.experiment_id = f"{timestamp}_{num_props}props"
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.epoch_metrics: List[EpochMetrics] = []
        self.eval_metrics: Dict[str, Any] = {}
        
        print(f"Experiment: {self.experiment_id}")
        print(f"Saving to: {self.experiment_dir}")
    
    def log_epoch(
        self,
        epoch: int,
        pred_loss: float,
        prop_loss: float,
        prop_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log metrics for a single epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            pred_loss: Prediction loss for epoch
            prop_loss: Property constraint loss for epoch
            prop_stats: Per-property statistics (e.g., constraint satisfaction)
        """
        if prop_stats is None:
            prop_stats = {}
        
        total_loss = pred_loss + self.config.lambda_prop * prop_loss
        
        metrics = EpochMetrics(
            epoch=epoch,
            pred_loss=pred_loss,
            prop_loss=prop_loss,
            total_loss=total_loss,
            prop_stats=prop_stats,
        )
        self.epoch_metrics.append(metrics)
    
    def log_evaluation(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        confusion_matrix: np.ndarray,
        classification_report: Dict[str, Dict[str, float]],
        test_prop_loss: Optional[float] = None,
        test_prop_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log evaluation metrics on test set.
        
        Args:
            accuracy: Overall accuracy
            precision: Macro-averaged precision
            recall: Macro-averaged recall
            f1: Macro-averaged F1 score
            confusion_matrix: Confusion matrix (array)
            classification_report: Scikit-learn classification report dict
            test_prop_loss: Average property loss on test set
            test_prop_stats: Property statistics on test set
        """
        self.eval_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix.tolist(),
            "classification_report": classification_report,
        }
        
        if test_prop_loss is not None:
            self.eval_metrics["test_prop_loss"] = test_prop_loss
        
        if test_prop_stats is not None:
            self.eval_metrics["test_prop_stats"] = test_prop_stats
    
    def save_confusion_matrix_image(self, figure) -> Path:
        """Save confusion matrix visualization to PNG.
        
        Args:
            figure: Matplotlib figure object with confusion matrix plot
            
        Returns:
            Path to saved PNG file
        """
        cm_path = self.experiment_dir / "confusion_matrix.png"
        figure.savefig(cm_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
        return cm_path
    
    def save_experiment(self) -> Path:
        """Save complete experiment config and results to JSON.
        
        Returns:
            Path to saved JSON file
        """
        results = {
            "config": self.config.to_dict(),
            "training": {
                "epoch_metrics": [asdict(m) for m in self.epoch_metrics],
                "final_pred_loss": (
                    self.epoch_metrics[-1].pred_loss if self.epoch_metrics else None
                ),
                "final_prop_loss": (
                    self.epoch_metrics[-1].prop_loss if self.epoch_metrics else None
                ),
                "final_total_loss": (
                    self.epoch_metrics[-1].total_loss if self.epoch_metrics else None
                ),
            },
            "evaluation": self.eval_metrics,
        }
        
        # Save to JSON
        results_path = self.experiment_dir / "config_and_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nExperiment saved to: {results_path}")
        return results_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this experiment run.
        
        Returns:
            Dictionary with key metrics and identifiers
        """
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.config.timestamp,
            "num_properties": len(self.config.properties),
            "property_names": [p.name for p in self.config.properties],
            "num_epochs": self.config.num_epochs,
            "final_accuracy": self.eval_metrics.get("accuracy"),
            "final_f1": self.eval_metrics.get("f1"),
            "final_pred_loss": (
                self.epoch_metrics[-1].pred_loss if self.epoch_metrics else None
            ),
            "final_prop_loss": (
                self.epoch_metrics[-1].prop_loss if self.epoch_metrics else None
            ),
        }
