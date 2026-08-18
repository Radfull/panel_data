from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import numpy as np

class PanelDataRequest(BaseModel):
    entities: List[str] = Field(..., min_items=2)
    time_periods: List[int] = Field(..., min_items=2)
    dependent_var: str = Field(..., min_length=1)
    independent_vars: List[str] = Field(..., min_items=1)
    data: Dict[str, List[List[float]]]
    
    @validator('data')
    def validate_data_shape(cls, v, values):
        if 'entities' in values and 'time_periods' in values:
            n_entities = len(values['entities'])
            n_periods = len(values['time_periods'])
            
            for var_name, var_data in v.items():
                if len(var_data) != n_entities:
                    raise ValueError(f"variable {var_name}: expected{n_entities} entities, got{len(var_data)}")
                for entity_data in var_data:
                    if len(entity_data) != n_periods:
                        raise ValueError(f"variable {var_name}: expected{n_periods} periods")
        return v

class ModelParamsRequest(BaseModel):
    entity_effects: bool = True
    time_effects: bool = False
    cov_type: str = "robust"

class EstimateModelRequest(BaseModel):
    data: PanelDataRequest
    params: ModelParamsRequest = ModelParamsRequest()

class CoefficientResponse(BaseModel):
    name: str
    value: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float

class ModelMetricsResponse(BaseModel):
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    nobs: int
    df_model: int
    df_residual: int

class EstimateModelResponse(BaseModel):
    coefficients: List[CoefficientResponse]
    metrics: ModelMetricsResponse
    fitted_values: List[float]
    residuals: List[float]
    model_serialized_base64: str