"""
amenonuboko API メインアプリケーション

このモジュールは、FastAPIアプリケーションのメインエントリーポイントです。
関連ブログタイトル抽出APIを提供します。
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi

from config import config
from app.routers import main_router, related_title, tags, my_events

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()


def custom_openapi():
    """
    カスタムOpenAPIスキーマを生成
    
    Returns:
        カスタムOpenAPIスキーマ
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="amenonuboko API",
        version="3.0.3",
        description="関連ブログタイトル抽出API",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TrustedHostミドルウェア（必要に応じて有効化）
# app.add_middleware(
#     TrustedHostMiddleware, 
#     allowed_hosts=["config.SITE_ROOT_URL"],
# )

# ルーターの登録
app.include_router(main_router.router)
app.include_router(related_title.router)
app.include_router(tags.router)
app.include_router(my_events.router)