from pydantic import BaseModel

from typing import Optional


class MessagesRequestModel(BaseModel):
    email: Optional[str] = "demo@example.com"
    question: str


class MessagesResponseModel(BaseModel):
    data: str
