from typing import List

from ninja import Schema


# key 는 파일 제목 역할을 할 변수입니다!
class TextRankResponse(Schema):
    keyword: List[str]