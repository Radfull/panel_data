from typing import Protocol
import base64
import numpy as np
from ..domain.entities import PanelData, ModelParams, ModelResult
from ..domain.errors import InsufficientDataError, InvalidDataError
from ..domain.ports import PanelModelEstimator
from .schemas import EstimateModelRequest, EstimateModelResponse

class PanelModelInteractor:
    
    def __init__(self, estimator: PanelModelEstimator):
        self._estimator = estimator
    
    def execute(self, request: EstimateModelRequest) -> EstimateModelResponse:
        data = self._to_domain_data(request)
        params = self._to_domain_params(request)
        
        self._validate(data, params)
        
        result = self._estimator.estimate(data, params)
        return self._to_response(result)
    
    def _to_domain_data(self, request: EstimateModelRequest) -> PanelData:
        values = {}
        for var_name, var_data in request.data.data.items():
            values[var_name] = np.array(var_data)
        
        return PanelData(
            entities=request.data.entities,
            time_periods=request.data.time_periods,
            dependent_var=request.data.dependent_var,
            independent_vars=request.data.independent_vars,
            values=values
        )
    
    def _to_domain_params(self, request: EstimateModelRequest) -> ModelParams:
        return ModelParams(
            entity_effects=request.params.entity_effects,
            time_effects=request.params.time_effects,
            cov_type=request.params.cov_type
        )
    
    def _validate(self, data: PanelData, params: ModelParams):
        min_obs = len(data.independent_vars) + 1
        if params.entity_effects:
            min_obs += data.n_entities - 1
        if params.time_effects:
            min_obs += data.n_periods - 1
        
        if data.n_entities * data.n_periods < min_obs:
            raise InsufficientDataError(
                f"need at least {min_obs} observations, got {data.n_entities * data.n_periods}"
            )
        
        for var_name in [data.dependent_var] + data.independent_vars:
            if var_name not in data.values:
                raise InvalidDataError(f"missing var: {var_name}")
            if np.isnan(data.values[var_name]).any():
                raise InvalidDataError(f"variable {var_name} contains nan vals")
    
    def _to_response(self, result: ModelResult) -> EstimateModelResponse:
        from .schemas import (
            CoefficientResponse, ModelMetricsResponse, 
            EstimateModelResponse
        )
        
        coefficients = [
            CoefficientResponse(
                name=c.name,
                value=c.value,
                std_error=c.std_error,
                t_stat=c.t_stat,
                p_value=c.p_value,
                ci_lower=c.ci_lower,
                ci_upper=c.ci_upper
            )
            for c in result.coefficients
        ]
        
        metrics = ModelMetricsResponse(
            r_squared=result.r_squared,
            adj_r_squared=result.adj_r_squared,
            f_statistic=result.f_statistic,
            f_p_value=result.f_p_value,
            nobs=result.nobs,
            df_model=result.df_model,
            df_residual=result.df_residual)
        
        return EstimateModelResponse(
            coefficients=coefficients,
            metrics=metrics,
            fitted_values=result.fitted_values.tolist(),
            residuals=result.residuals.tolist(),
            model_serialized_base64=base64.b64encode(result.model_serialized).decode('utf-8')
        )

    
    