from django.http import HttpRequest
from ninja import Router
from ninja import Form
from TextRankapp.services.TR_service import textrank_keyword,komoran_tokenize
from .schemas import TextRankRequest,TextRankResponse


router = Router()

@router.post("/",response=TextRankResponse)
def nst(request: HttpRequest, TR_request: TextRankRequest = Form(...)) -> dict:
    keyword = textrank_keyword(TR_request.chat_log,komoran_tokenize, 2,2,2)
    return {"keyword": keyword}