from fastapi import APIRouter

from fastapi.responses import JSONResponse
from app.db.sqlalchemy.crud import crud 
import json
import requests

import config


router = APIRouter()

@router.get("/amenonuboko/v1/myEvents/")
async def read_root():
    return json.dumps(["パスパラメーターに[年月(例:202210)]を指定してください！"], indent=2, ensure_ascii=False)


@router.get("/amenonuboko/v1/myEvents/{ym}")
async def read_item(ym: str):
    if (ym == None):
        return json.dumps(["パラメータがありません！"], indent=2, ensure_ascii=False)

    url = config.CONNPASS_URL +  "/?nickname=" + config.CONNPASS_NICKNAME +"&ym=" + ym

    try:
        res = requests.get(url)
        res.raise_for_status()

        data = res.json()

    except requests.exceptions.RequestException as e:
        print("エラー : ", e)
        return [str(e)]

    return JSONResponse(data)