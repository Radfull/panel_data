from abc import ABC, abstractmethod
from .entities import PanelData, ModelParams, ModelResult

class PanelModelEstimator(ABC):
    
    @abstractmethod
    def estimate(self, data: PanelData, params: ModelParams) ->ModelResult:
        pass