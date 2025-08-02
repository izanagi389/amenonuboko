"""
メインルーターモジュール

このモジュールは、アプリケーションのメインエンドポイントと
スケジュールタスクを提供します。
"""

import logging

from fastapi import APIRouter, BackgroundTasks
from fastapi_utils.tasks import repeat_every

from app.scheduler import init as scheduler_init
from app.utils.response import create_success_response

router = APIRouter()


@router.on_event("startup")
@repeat_every(seconds=60 * 60 * 24)  # 24時間ごと
def startup_event() -> None:
    """
    アプリケーション起動時のスケジュールタスク
    
    24時間ごとにスケジューラータスクを実行します。
    """
    logging.info('スケジューラータスクを開始します')
    try:
        BackgroundTasks().add_task(scheduler_init())
    except Exception as e:
        logging.error(f"スケジューラータスクの実行に失敗しました: {e}")


@router.get("/amenonuboko/")
async def read_root():
    """
    メインエンドポイント
    
    Returns:
        アプリケーション情報とエンドポイント一覧
    """
    return create_success_response({
        "message": "こちらは関連のブログタイトル抽出APIです。",
        "version": "2.0",
        "endpoints": {
            "related_title": "/amenonuboko/v2/related_title/{blog_id}",
            "tags": "/amenonuboko/v1/tags/{blog_id}",
            "my_events": "/amenonuboko/v1/myEvents/{year_month}"
        }
    })


@router.get("/amenonuboko/health")
async def health_check():
    """
    ヘルスチェックエンドポイント
    
    Returns:
        サービス状態情報
    """
    return create_success_response({
        "status": "healthy",
        "service": "amenonuboko-api"
    })
