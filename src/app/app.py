import uvicorn
from fastapi import FastAPI
from app.endpoints import router


def create_app() -> FastAPI:
    app = FastAPI(title="rev_analyzer API", version="0.0.1")
    app.include_router(router)    
    return app


def start_app():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
