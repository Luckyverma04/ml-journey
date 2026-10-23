from fastapi import FastAPI
from routers.verify import router as verify_router

app = FastAPI()
app.include_router(verify_router)

@app.get("/health")
def health():
    return {"status": "ok"}