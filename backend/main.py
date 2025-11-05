from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, shutil, time

from .algorithms import lsb, lsb_matching, dct, dwt, dwt_dct_hybrid
from .metrics import get_metrics

app = FastAPI(title="Steganography API")

allow_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]


app.add_middleware(
    CORSMiddleware,
    # allow_origins=allow_origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure directory exists
os.makedirs("results/outputs", exist_ok=True)

# Serve static files from the outputs folder
app.mount("/download", StaticFiles(directory="results/outputs"), name="download")

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, "..", "results", "uploads")
OUTPUT_DIR = os.path.join(ROOT, "..", "results", "outputs")
EXTRACT_DIR = os.path.join(ROOT, "..", "results", "extracted")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

ALGOS = {
    "lsb": lsb,
    "lsb_matching": lsb_matching,
    "dct": dct,
    "dwt": dwt,
    "hybrid": dwt_dct_hybrid
}

@app.post("/embed/")
async def embed(file: UploadFile = File(...), message: str = Form(...), algorithm: str = Form(...)):
    if algorithm not in ALGOS:
        return JSONResponse({"error": "Invalid algorithm"}, status_code=400)

    src = os.path.join(UPLOAD_DIR, file.filename)
    with open(src, "wb") as f: shutil.copyfileobj(file.file, f)
    name, ext = os.path.splitext(file.filename)
    out_name = f"stego_{algorithm}_{name}{ext}"
    dst = os.path.join(OUTPUT_DIR, out_name)

    start = time.time()
    ALGOS[algorithm].embed_message(src, message, dst)
    end = time.time()
    metrics = get_metrics(src, dst, start, end)
    metrics["Algorithm"] = algorithm
    return {"metrics": metrics, "output_path": f"/download/{out_name}"}

@app.post("/extract/")
async def extract(file: UploadFile = File(...), algorithm: str = Form(...)):
    if algorithm not in ALGOS:
        return JSONResponse({"error": "Invalid algorithm"}, status_code=400)
    src = os.path.join(UPLOAD_DIR, file.filename)
    with open(src, "wb") as f: shutil.copyfileobj(file.file, f)
    msg = ALGOS[algorithm].extract_message(src)
    return {"algorithm": algorithm, "message": msg}

@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename)
    return JSONResponse({"error": "Not found"}, status_code=404)
