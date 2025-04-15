from pydantic import BaseModel
# from fastapi import UploadFile, File, Form
from typing import Optional


class MessagesRequestModel(BaseModel):
    email: Optional[str] = "demo@example.com"
    question: str
    use_llm: Optional[bool] = False


class MessagesResponseModel(BaseModel):
    data: str
    error: Optional[str] = ""


# class BulkMessagesRequestModel(BaseModel):
#     file: UploadFile = File(...)
#     email: Optional[str] = Form("demo@example.com")


class BulkMessagesResponseModel(BaseModel):
    success: bool
    processedCsvData: str
