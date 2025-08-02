"""
マイイベントルーターモジュール

このモジュールは、Connpass APIを使用したイベントデータ取得のための
APIエンドポイントを提供します。
"""

import logging

import requests
from fastapi import APIRouter

import config
from app.utils.response import create_error_response, create_success_response, validate_parameter

# 設定値
CONNPASS_CONFIG = {
    'url': config.CONNPASS_URL,
    'nickname': config.CONNPASS_NICKNAME
}

router = APIRouter()


@router.get("/amenonuboko/v1/myEvents/")
async def read_root():
    """
    パラメータなしのエンドポイント
    
    Returns:
        エラーレスポンス（年月が指定されていない場合）
    """
    return create_error_response("パスパラメーターに[年月(例:202210)]を指定してください！")


@router.get("/amenonuboko/v1/myEvents/{ym}")
async def read_item(ym: str):
    """
    イベントデータを取得
    
    Args:
        ym: 年月（YYYYMM形式）
        
    Returns:
        イベントデータまたはエラーレスポンス
    """
    # パラメータバリデーション
    error_response = validate_parameter(ym, "年月")
    if error_response:
        return error_response
    
    # 年月形式のバリデーション（YYYYMM形式）
    if not ym.isdigit() or len(ym) != 6:
        return create_error_response("年月は6桁の数字（例:202210）で指定してください")
    
    try:
        # URLを構築
        url = f"{CONNPASS_CONFIG['url']}?nickname={CONNPASS_CONFIG['nickname']}&ym={ym}"
        
        # リクエスト実行
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return create_success_response(data)
        
    except requests.exceptions.Timeout:
        return create_error_response("リクエストがタイムアウトしました", 408)
    except requests.exceptions.RequestException as e:
        logging.error(f"Connpass API エラー: {e}")
        return create_error_response(f"外部APIの取得に失敗しました: {str(e)}", 502)
    except Exception as e:
        logging.error(f"予期しないエラー: {e}")
        return create_error_response(f"データの取得に失敗しました: {str(e)}", 500)