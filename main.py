import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mode_detector import ModeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harmonic-pro")

app = FastAPI(title="Harmonic Pro DJ API")

@app.get("/health")
def health():
    return {"status": "ok", "service": "harmonic-pro"}

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ModeDetector()

@app.get("/")
async def health_check():
    return {"status": "online", "service": "Harmonic Pro API", "version": "1.0.0"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    logger.info(f"Analyzing: {file.filename}")

    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported format. Use MP3, WAV, FLAC, OGG or M4A.")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = detector.analyze(tmp_path)

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis error")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
