"""Property-based constraints for neural network verification.

This module provides a framework for defining and managing properties as soft constraints
that guide model training. Properties are represented as classes that compute soft
constraints, enabling extensibility for new constraint types.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def soft_leq(x: torch.Tensor, threshold: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
    """Compute a soft less-than-or-equal constraint using sigmoid.
    
    Args:
        x: Values to compare (shape: any)
        threshold: Threshold value(s)
        sharpness: Controls sigmoid steepness (higher = sharper, more like hard constraint)
    
    Returns:
        Soft constraint values in [0, 1] (shape: same as x)
    """
    return torch.sigmoid(sharpness * (threshold - x))


class Property:
    """Base class for property-based constraints.
    
    A property defines a constraint on network inputs and their relationship to
    predicted class probabilities. Properties are used to enforce domain knowledge
    during training.
    """
    
    def __init__(
        self,
        name: str,
        feature_name: str,
        feature_idx: int,
        theta: float,
        theta_scaled: float,
        sharpness: float = 10.0,
    ):
        """Initialize a property.
        
        Args:
            name: Display name for this property (e.g., "LowBytes")
            feature_name: Name of the feature this property constrains (e.g., "orig_bytes")
            feature_idx: Index of feature in the feature vector
            theta: Original (unscaled) threshold value
            theta_scaled: Scaled threshold value (after StandardScaler transformation)
            sharpness: Sigmoid sharpness parameter for soft constraint
        """
        self.name = name
        self.feature_name = feature_name
        self.feature_idx = feature_idx
        self.theta = theta
        self.theta_scaled = theta_scaled
        self.sharpness = sharpness
    
    def soft_constraint(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Compute soft constraint for this property.
        
        Args:
            x_flat: Feature tensor of shape (N, F) where F is number of features
        
        Returns:
            Soft constraint values of shape (N,) in range [0, 1]
        """
        return soft_leq(x_flat[:, self.feature_idx], self.theta_scaled, self.sharpness)
    
    def __repr__(self) -> str:
        return f"{self.name}(feature={self.feature_name}, idx={self.feature_idx}, theta={self.theta})"


class LowBytesProperty(Property):
    """Property: Low original bytes constraint.
    
    Represents: "Connections with low original bytes are likely benign"
    """
    pass


class LowPacketsProperty(Property):
    """Property: Low original packets constraint.
    
    Represents: "Connections with low original packets are likely benign"
    """
    pass


class PropertyCollection:
    """Manager for multiple properties and their combined loss computation.
    
    This class aggregates multiple properties and computes their combined
    loss as a conjunction (AND logic): all properties must be satisfied simultaneously.
    """
    
    def __init__(self, properties: List[Property]):
        """Initialize property collection.
        
        Args:
            properties: List of Property objects
        """
        self.properties = properties
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        p_dos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute property-based loss.
        
        The loss encourages: if (all properties are satisfied) then predict benign (not DOS).
        This is implemented as: loss = E[AND(properties) * (1 - p_dos)]
        
        Args:
            model: Neural network model
            x: Input batch of shape (N, 1, F)
            p_dos: Probability of DOS attack of shape (N,)
        
        Returns:
            Tuple of (loss, stats_dict) where:
                - loss is a scalar tensor
                - stats_dict contains per-property and aggregate statistics
        """
        x_flat = x[:, 0, :]  # Extract feature dimension (N, F)
        
        # Compute soft constraint for each property
        constraints = []
        for prop in self.properties:
            constraint = prop.soft_constraint(x_flat)
            constraints.append(constraint)
        
        # AND logic: minimum across properties
        if len(constraints) == 1:
            antecedent = constraints[0]
        else:
            # Stack and take minimum across properties axis
            antecedent = torch.stack(constraints, dim=1).min(dim=1)[0]
        
        # Loss: if antecedent (all properties satisfied) then predict benign (low DOS prob)
        loss = torch.mean(antecedent * (1.0 - p_dos))
        
        # Collect statistics
        stats = {"antecedent": antecedent.mean().item(), "p_dos": p_dos.mean().item()}
        for i, (prop, constraint) in enumerate(zip(self.properties, constraints)):
            stats[prop.name] = constraint.mean().item()
        
        return loss, stats
    
    def __repr__(self) -> str:
        props_str = "\n  ".join(repr(p) for p in self.properties)
        return f"PropertyCollection[\n  {props_str}\n]"
