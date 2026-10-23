import sys
import cv2
from services.matcher import get_embedding, score_pair, zone

path_a, path_b = sys.argv[1], sys.argv[2]
img_a, img_b = cv2.imread(path_a), cv2.imread(path_b)

if img_a is None or img_b is None:
    raise SystemExit("koi photo khuli nahi, file ka naam ya rasta check kar")

a, na = get_embedding(img_a)
b, nb = get_embedding(img_b)
print("chehre:", na, nb)

if a is None or b is None:
    raise SystemExit("kisi photo mein chehra nahi mila")

s = score_pair(a, b)
print("score:", round(s, 3), "|", "HUMAN_REVIEW" if (na > 1 or nb > 1) else zone(s))