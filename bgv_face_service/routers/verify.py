import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas import VerifyRequest
from services.image_loader import load_image_from_url
from services.matcher import get_embedding, score_pair, zone

router = APIRouter()
MAX_CANDIDATES = 15
MAX_BYTES = 5_000_000


@router.post("/verify")
def verify(req: VerifyRequest):
    if len(req.candidate_urls) > MAX_CANDIDATES:
        raise HTTPException(status_code=400, detail="max 15 URLs allowed")

    try:
        ref_img = load_image_from_url(req.reference_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"reference photo: {e}")

    ref_emb, ref_faces = get_embedding(ref_img)
    if ref_emb is None:
        raise HTTPException(status_code=400, detail="reference photo mein chehra nahi mila")
    if ref_faces > 1:
        raise HTTPException(status_code=400, detail="reference photo mein 2+ chehre")

    results = []
    for url in req.candidate_urls:
        try:
            img = load_image_from_url(url)
        except Exception as e:
            results.append({"url": url, "status": "failed", "reason": str(e)})
            continue

        emb, n = get_embedding(img)
        if emb is None:
            results.append({"url": url, "status": "no_face"})
            continue

        s = score_pair(ref_emb, emb)
        z = "HUMAN_REVIEW" if n > 1 else zone(s)
        results.append({"url": url, "status": "ok", "faces": n,
                        "score": round(s, 3), "zone": z})

    return {"results": results}


def _read_upload(f: UploadFile):
    data = f.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="file bahut badi hai")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="ye image nahi hai")
    return img


@router.post("/verify-upload")
def verify_upload(reference_url: str = Form(...), candidate: UploadFile = File(...)):
    try:
        ref_img = load_image_from_url(reference_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"reference photo: {e}")
    cand_img = _read_upload(candidate)

    a, na = get_embedding(ref_img)
    b, nb = get_embedding(cand_img)
    if a is None or b is None:
        return {"status": "no_face", "faces": [na, nb]}

    s = score_pair(a, b)
    return {"status": "ok", "faces": [na, nb], "score": round(s, 3), "zone": zone(s)}