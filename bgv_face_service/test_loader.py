from services.image_loader import load_image_from_url

for u in ["http://example.com/a.jpg", "https://localhost/a.jpg", "https://127.0.0.1/a.jpg"]:
    try:
        load_image_from_url(u)
    except Exception as e:
        print(u, "->", e)