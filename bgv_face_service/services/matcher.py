from pathlib import Path
import cv2

MODELS = Path(__file__).resolve().parent.parent / "models"

_recognizer = cv2.FaceRecognizerSF.create(str(MODELS / "sface.onnx"), "")

ACCEPT_AT = 0.5
REVIEW_AT = 0.363


def get_embedding(img):
    h, w = img.shape[:2]
    if w > 640:
        img = cv2.resize(img, (640, int(h * 640 / w)))
        h, w = img.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (w, h), 0.8)
    _, faces = detector.detect(img)
    if faces is None:
        return None, 0
    aligned = _recognizer.alignCrop(img, faces[0])
    return _recognizer.feature(aligned), len(faces)


def score_pair(emb_a, emb_b):
    return float(_recognizer.match(emb_a, emb_b, cv2.FaceRecognizerSF_FR_COSINE))


def zone(score):
    if score >= ACCEPT_AT:
        return "ACCEPT"
    if score >= REVIEW_AT:
        return "HUMAN_REVIEW"
    return "REJECT"