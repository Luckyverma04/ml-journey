from pydantic import BaseModel

class VerifyRequest(BaseModel):
    reference_url: str
    candidate_urls: list[str]