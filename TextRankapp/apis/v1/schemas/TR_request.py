from ninja import Schema
from ninja import Schema
from typing import List


# key 는 파일 제목 역할을 할 변수입니다!
class TextRankRequest(Schema):
    chat_log: List[str]
