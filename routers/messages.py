from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from icecream import ic
from io import StringIO

from routers.models.messages import MessagesRequestModel, MessagesResponseModel, BulkMessagesResponseModel
from services.messages import messages_service

router = APIRouter()


@router.post(
    '',
    response_model=MessagesResponseModel
)
def list_messages(data: MessagesRequestModel):

    res = messages_service.predict_scores(data.question, data.email, data.use_llm)
    return {
        'data': res
    }


@router.post(
    '/bulk',
    response_model=BulkMessagesResponseModel
)
async def list_messages_csv(
    file: UploadFile = File(...),
    email: str = Form("demo@example.com")
):
    """
    Process multiple messages from a CSV file and return predictions in JSON format
    """
    try:
        csv_data = await messages_service.process_csv_file(file, email)
        ic("success")
        return BulkMessagesResponseModel(
            success=True,
            processedCsvData=csv_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing CSV file")
