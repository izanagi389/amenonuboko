from fastapi import APIRouter
from fastapi.responses import JSONResponse
import json

from app.lib.related_score import create_score


router = APIRouter()


@router.get("/amenonuboko/v2/related_title/")
async def read_root():
    return json.dumps(["パスパラメーターにブログIDを指定してください！"], indent=2, ensure_ascii=False)


@router.get("/amenonuboko/v2/related_title/{search_id}")
async def read_item(search_id: str):
    if (search_id == None):
        return json.dumps(["パラメータがありません！"], indent=2, ensure_ascii=False)

    return JSONResponse(create_score().get_scores(5, search_id))
