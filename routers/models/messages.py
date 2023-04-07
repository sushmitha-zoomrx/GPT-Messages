from typing import List, Union

from pydantic import BaseModel


class MessagesRequestModel(BaseModel):
    email: str
    question: str


class MessagesResponseModel(BaseModel):
    data: str
