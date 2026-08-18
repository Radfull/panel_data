from fastapi import FastAPI
from .presentation.router import router

app = FastAPI(
    version="1.0.0"
)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}