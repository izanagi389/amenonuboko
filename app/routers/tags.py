"""
タグルーターモジュール

このモジュールは、タグデータ取得のためのAPIエンドポイントを提供します。
"""

from fastapi import APIRouter

from app.db.sqlalchemy.crud import crud
from app.utils.response import create_error_response, create_success_response, validate_parameter

router = APIRouter()


@router.get("/amenonuboko/v1/tags/")
async def read_root():
    """
    パラメータなしのエンドポイント
    
    Returns:
        エラーレスポンス（ブログIDが指定されていない場合）
    """
    return create_error_response("パスパラメーターにブログIDを指定してください！")


@router.get("/amenonuboko/v1/tags/{blog_id}")
async def read_item(blog_id: str):
    """
    タグデータを取得
    
    Args:
        blog_id: 検索対象のブログID
        
    Returns:
        タグデータまたはエラーレスポンス
    """
    # パラメータバリデーション
    error_response = validate_parameter(blog_id, "ブログID")
    if error_response:
        return error_response
    
    try:
        # トピックデータを取得
        result = crud().get_topic_data_for_df(blog_id)
        
        # エラーレスポンスの場合はそのまま返す
        if "error" in result:
            return create_error_response(result["error"], 404)
        
        return create_success_response(result)
    except Exception as e:
        return create_error_response(f"データの取得に失敗しました: {str(e)}", 500)