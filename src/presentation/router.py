from fastapi import APIRouter, Depends, HTTPException
from ..application.interactor import PanelModelInteractor
from ..application.schemas import EstimateModelRequest, EstimateModelResponse
from ..domain.errors import InsufficientDataError, InvalidDataError, ModelEstimationError
from .dependencies import get_interactor

router = APIRouter(prefix="/api/v1", tags=["panel-models"])

@router.post("/estimate", response_model=EstimateModelResponse)
async def estimate_model(
    request: EstimateModelRequest,
    interactor: PanelModelInteractor = Depends(get_interactor)
) -> EstimateModelResponse:
    try:
        result = interactor.execute(request)
        return result
    except InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except InvalidDataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelEstimationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal server error: {str(e)}")