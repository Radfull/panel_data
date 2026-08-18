from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass(frozen=True)
class PanelData:
    entities: List[str]
    time_periods: List[int]
    dependent_var: str
    independent_vars: List[str]
    values: Dict[str, np.ndarray]
    
    @property
    def n_entities(self) -> int:
        return len(self.entities)
    
    @property
    def n_periods(self) -> int:
        return len(self.time_periods)

@dataclass(frozen=True)
class ModelParams:
    entity_effects: bool = True
    time_effects: bool = False
    cov_type: str = "robust" 
    
    def __post_init__(self):
        if self.cov_type not in ["robust", "clustered", "nonrobust"]:
            raise ValueError(f"Invalid cov_type: {self.cov_type}")

@dataclass
class CoefficientEstimate:
    name: str
    value: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float

@dataclass
class ModelResult:
    coefficients: List[CoefficientEstimate]
    fitted_values:np.ndarray
    residuals: np.ndarray
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    model_serialized: bytes
    nobs: int
    df_model: int
    df_residual: int