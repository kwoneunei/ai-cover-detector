from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.inference import SingGraphInference

app = FastAPI(title="AI Cover Detection API")

MODEL = SingGraphInference(
    config_path="SingGraph/utils/SingGraph.conf",
    weight_path="/path/to/best.pth",
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file format"}
        )

    file_bytes = await file.read()
    result = MODEL.predict_bytes(file_bytes)
    return result
