"""
アプリケーション設定モジュール

このモジュールは、アプリケーション全体で使用される設定を管理します。
環境変数から値を取得し、デフォルト値を提供します。
"""

import os
from typing import List
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# .envファイルが存在しない場合の警告
if not os.path.exists('.env'):
    print("⚠️  .env ファイルが見つかりません。env.example をコピーして設定してください。")
    print("cp env.example .env")


class Config:
    """アプリケーション設定クラス"""
    
    # CORS設定
    ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com"
    ]
    
    # データベース設定
    MYSQL_USER: str = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD")
    MYSQL_HOST: str = os.getenv("MYSQL_HOST")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE")
    MYSQL_PORT: str = os.getenv("MYSQL_PORT")
    
    # MicroCMS設定
    MICROCMS_URL: str = os.getenv("MICROCMS_URL")
    MICROCMS_API_KEY: str = os.getenv("MICROCMS_API_KEY")
    LIMIT: str = os.getenv("LIMIT", "100")
    
    # Connpass設定
    CONNPASS_URL: str = os.getenv("CONNPASS_URL")
    CONNPASS_NICKNAME: str = os.getenv("CONNPASS_NICKNAME")
    
    # サイト設定
    SITE_ROOT_URL: str = os.getenv("SITE_ROOT_URL")


# グローバル設定インスタンス
config = Config() 