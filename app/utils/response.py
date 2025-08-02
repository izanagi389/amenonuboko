"""
レスポンスユーティリティモジュール

このモジュールは、FastAPIアプリケーションで使用する
共通のレスポンス生成機能を提供します。
"""

import json
from typing import Any, Dict, List, Optional

from fastapi.responses import JSONResponse


def create_error_response(message: str, status_code: int = 400) -> JSONResponse:
    """
    共通のエラーレスポンスを作成
    
    Args:
        message: エラーメッセージ
        status_code: HTTPステータスコード
        
    Returns:
        エラーレスポンス
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": message}
    )


def create_success_response(data: Any, status_code: int = 200) -> JSONResponse:
    """
    共通の成功レスポンスを作成
    
    Args:
        data: レスポンスデータ
        status_code: HTTPステータスコード
        
    Returns:
        成功レスポンス
    """
    return JSONResponse(
        status_code=status_code,
        content=data
    )


def create_json_response(data: List[str], indent: int = 2) -> str:
    """
    JSON文字列レスポンスを作成（既存の形式との互換性のため）
    
    Args:
        data: レスポンスデータ
        indent: JSONインデント
        
    Returns:
        JSON文字列
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def validate_parameter(param: Optional[str], param_name: str = "parameter") -> Optional[JSONResponse]:
    """
    パラメータのバリデーション
    
    Args:
        param: バリデーション対象のパラメータ
        param_name: パラメータ名
        
    Returns:
        バリデーションエラーの場合エラーレスポンス、成功時None
    """
    if param is None or param.strip() == "":
        return create_error_response(f"{param_name}が指定されていません！")
    return None 