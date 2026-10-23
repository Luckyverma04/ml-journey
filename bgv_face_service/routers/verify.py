from fastapi import APIRouter, HTTPException
from schemas import VerifyRequest
from services.image_loader import load_image_from_url
from services.matcher import get_embedding, score_pair, zone

router = APIRouter()
MAX_CANDIDATES = 15


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