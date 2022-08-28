from fastapi import APIRouter

from fastapi.responses import JSONResponse
from app.db.sqlalchemy.crud import crud 
import json


router = APIRouter()

@router.get("/amenonuboko/v1/tags/")
async def read_root():
    return json.dumps(["パスパラメーターにブログIDを指定してください！"], indent=2, ensure_ascii=False)


@router.get("/amenonuboko/v1/tags/{search_id}")
async def read_item(search_id: str):
    if (search_id == None):
        return json.dumps(["パラメータがありません！"], indent=2, ensure_ascii=False)

    return JSONResponse(crud().get_topic_data_for_df(search_id))