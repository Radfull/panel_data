from functools import lru_cache
from ..application.interactor import PanelModelInteractor
from ..infrastructure.statsmodels_adapter import StatsmodelsPanelAdapter

@lru_cache()
def get_interactor() -> PanelModelInteractor:
    estimator = StatsmodelsPanelAdapter()
    return PanelModelInteractor(estimator)