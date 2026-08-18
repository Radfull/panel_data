import pickle
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from ..domain.entities import PanelData, ModelParams, ModelResult, CoefficientEstimate
from ..domain.ports import PanelModelEstimator
from ..domain.errors import ModelEstimationError

class StatsmodelsPanelAdapter(PanelModelEstimator):
    
    def estimate(self, data: PanelData, params: ModelParams) -> ModelResult:
        try:
            df = self._prepare_dataframe(data)
            
            if params.entity_effects and params.time_effects:
                model = PanelOLS(
                    df[data.dependent_var],
                    df[data.independent_vars],
                    entity_effects=True,
                    time_effects=True
                )
            elif params.entity_effects:
                model = PanelOLS(
                    df[data.dependent_var],
                    df[data.independent_vars],
                    entity_effects=True,
                    time_effects=False
                )
            elif params.time_effects:
                model = PanelOLS(
                    df[data.dependent_var],
                    df[data.independent_vars],
                    entity_effects=False,
                    time_effects=True
                )
            else:
                model = RandomEffects(
                    df[data.dependent_var],
                    df[data.independent_vars]
                )
            
            results = model.fit(cov_type=params.cov_type)
            
            coefficients = []
            for name in results.params.index:
                ci = results.conf_int()
                coefficients.append(CoefficientEstimate(
                    name=name,
                    value=results.params[name],
                    std_error=results.std_errors[name],
                    t_stat=results.tstats[name],
                    p_value=results.pvalues[name],
                    ci_lower=ci.loc[name, 'lower'],
                    ci_upper=ci.loc[name, 'upper']
                ))
            
            model_bytes = pickle.dumps(results)
            
            return ModelResult(
                coefficients=coefficients,
                fitted_values=results.fitted_values.values.flatten(),
                residuals=results.resids.values.flatten(),
                r_squared=results.rsquared,
                adj_r_squared=results.rsquared,
                f_statistic=results.f_statistic.stat,
                f_p_value=results.f_statistic.pval,
                model_serialized=model_bytes,
                nobs=results.nobs,
                df_model=results.df_model,
                df_residual=results.df_resid
            )
            
        except Exception as e:
            raise ModelEstimationError(f"failed to estimate model:{str(e)}")
    
    def _prepare_dataframe(self, data: PanelData) -> pd.DataFrame:

        index = pd.MultiIndex.from_product(
            [data.entities, data.time_periods],
            names=['entity', 'time'])
        
        df_dict = {}
        for var_name, var_data in data.values.items():
            df_dict[var_name] = var_data.flatten()
        
        df = pd.DataFrame(df_dict, index=index)
        return df