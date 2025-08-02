"""
関連タイトルルーターモジュール

このモジュールは、関連タイトル取得のためのAPIエンドポイントを提供します。
"""

from fastapi import APIRouter, HTTPException

from app.lib.related_score import create_score
from app.utils.response import create_error_response, create_success_response, validate_parameter

router = APIRouter()


@router.get("/amenonuboko/v2/related_title/")
async def read_root():
    """
    パラメータなしのエンドポイント
    
    Returns:
        エラーレスポンス（ブログIDが指定されていない場合）
    """
    return create_error_response("パスパラメーターにブログIDを指定してください！")


@router.get("/amenonuboko/v2/related_title/{search_id}")
async def read_item(search_id: str):
    """
    関連タイトルを取得
    
    Args:
        search_id: 検索対象のブログID
        
    Returns:
        関連タイトルのリストまたはエラーレスポンス
    """
    # パラメータバリデーション
    error_response = validate_parameter(search_id, "ブログID")
    if error_response:
        return error_response
    
    try:
        # 関連スコアを取得
        result = create_score().get_scores(5, search_id)
        return create_success_response(result)
    except Exception as e:
        return create_error_response(f"データの取得に失敗しました: {str(e)}", 500)
