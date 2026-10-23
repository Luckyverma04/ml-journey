import ipaddress
import socket
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np

MAX_BYTES = 5_000_000
TIMEOUT = 10


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _check_url(url):
    p = urlparse(url)
    if p.scheme != "https":
        raise ValueError("sirf https allowed")
    if not p.hostname:
        raise ValueError("URL galat hai")
    for info in socket.getaddrinfo(p.hostname, 443):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("private address allowed nahi")


def load_image_from_url(url):
    _check_url(url)
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "bgv-face-service"})
    with opener.open(req, timeout=TIMEOUT) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("file bahut badi hai")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("ye image nahi hai")
    return img