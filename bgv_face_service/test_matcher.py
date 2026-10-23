import cv2
from services.matcher import get_embedding, score_pair, zone

a, na = get_embedding(cv2.imread("../test2.png"))
b, nb = get_embedding(cv2.imread("../test5.JPEG"))
s = score_pair(a, b)
print("chehre:", na, nb, "| score:", round(s, 3), "|", zone(s))