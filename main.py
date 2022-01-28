
import config

from fastapi import FastAPI, BackgroundTasks
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

from app import crud, scheduler


app = FastAPI()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="3.0.3",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init(background_tasks: BackgroundTasks):
    background_tasks.add_task(
        scheduler.scheduler.start(), message="some notification")


@app.get("/api/")
def read_root():
    return json.dumps(["こちらは関連のブログタイトル抽出APIです。"], indent=2, ensure_ascii=False)


@app.get("/api/v1/")
def read_root():
    return json.dumps(["パスが間違っています！"], indent=2, ensure_ascii=False)


@app.get("/api/v1/related_title/")
def read_root():
    return json.dumps(["パスパラメーターにブログIDを指定してください！"], indent=2, ensure_ascii=False)


@app.get("/api/v1/related_title/{search_id}")
def read_item(search_id: str):
    print(search_id)
    if (search_id == None):
        return json.dumps(["パラメータがありません！"], indent=2, ensure_ascii=False)

    return JSONResponse(crud.get_related_titles(search_id))