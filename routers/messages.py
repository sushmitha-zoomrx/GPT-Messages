from fastapi import APIRouter
from icecream import ic

from routers.models.messages import MessagesRequestModel, MessagesResponseModel
from services.messages import messages_service

router = APIRouter()


@router.post(
    '',
    response_model=MessagesResponseModel
)
def list_messages(data: MessagesRequestModel):

    res = messages_service.predict_scores(data.question)
    return {
        'data': res
    }
