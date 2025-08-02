"""
アプリケーション設定モジュール

このモジュールは、アプリケーション全体で使用される設定を管理します。
環境変数から値を取得し、デフォルト値を提供します。
"""

import os
from typing import List


class Config:
    """アプリケーション設定クラス"""
    
    # CORS設定
    ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com"
    ]
    
    # データベース設定
    MYSQL_USER: str = os.getenv("MYSQL_USER", "amenonuboko_user")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "amenonuboko_password")
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "amenonuboko")
    MYSQL_PORT: str = os.getenv("MYSQL_PORT", "3306")
    
    # MicroCMS設定
    MICROCMS_URL: str = os.getenv("MICROCMS_URL", "https://your-microcms.microcms.io/api/v1/contents")
    MICROCMS_API_KEY: str = os.getenv("MICROCMS_API_KEY", "your_microcms_api_key")
    LIMIT: str = os.getenv("LIMIT", "100")
    
    # Connpass設定
    CONNPASS_URL: str = os.getenv("CONNPASS_URL", "https://connpass.com/api/v1/event/")
    CONNPASS_NICKNAME: str = os.getenv("CONNPASS_NICKNAME", "your_connpass_nickname")
    
    # サイト設定
    SITE_ROOT_URL: str = os.getenv("SITE_ROOT_URL", "http://localhost:8000")


# グローバル設定インスタンス
config = Config() 