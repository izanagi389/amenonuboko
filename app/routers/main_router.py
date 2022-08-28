from fastapi import APIRouter, BackgroundTasks
from fastapi_utils.tasks import repeat_every
import json

from app import scheduler


router = APIRouter()

@router.on_event("startup")
@repeat_every(seconds=60 * 60 * 24)
def startup_event():
    print('Start')
    BackgroundTasks().add_task(scheduler.init())


@router.get("/amenonuboko/")
async def read_root():
    return json.dumps(["こちらは関連のブログタイトル抽出APIです。"], indent=2, ensure_ascii=False)


@router.get("/amenonuboko/v1/")
async def read_root():
    return json.dumps(["パスが間違っています！"], indent=2, ensure_ascii=False)

@router.get("/amenonuboko/v2/")
async def read_root():
    return json.dumps(["パスが間違っています！"], indent=2, ensure_ascii=False)
