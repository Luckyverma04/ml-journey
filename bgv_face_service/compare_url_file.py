import sys
import cv2
import matplotlib.pyplot as plt
from services.image_loader import load_image_from_url
from services.matcher import get_embedding, score_pair, zone

url, path = sys.argv[1], sys.argv[2]

img_url = load_image_from_url(url)
img_file = cv2.imread(path)
if img_file is None:
    raise SystemExit("local file khuli nahi, naam ya rasta check kar")

a, na = get_embedding(img_url)
b, nb = get_embedding(img_file)
print("chehre (URL, file):", na, nb)

if a is None or b is None:
    raise SystemExit("kisi photo mein chehra nahi mila")

s = score_pair(a, b)
z = zone(s)
print("score:", round(s, 3), "|", z)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img_url, cv2.COLOR_BGR2RGB))
axes[0].set_title("URL wali photo")
axes[1].imshow(cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB))
axes[1].set_title("local file")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"similarity: {s:.3f}  →  {z}")
plt.show()