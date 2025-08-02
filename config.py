"""
アプリケーション設定モジュール

このモジュールは、アプリケーション全体で使用される設定を管理します。
環境変数から値を取得し、デフォルト値を提供します。
"""

import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# プロジェクトルートディレクトリを取得
BASE_DIR = Path(__file__).resolve().parent

# .envファイルのパスを指定して読み込み
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# .envファイルが存在しない場合の警告（Docker環境では環境変数が既に設定されているためスキップ）
if not env_path.exists() and not os.getenv("MYSQL_USER"):
    print("⚠️  .env ファイルが見つかりません。")
    print(f"   プロジェクトルートディレクトリに .env ファイルを作成してください。")
    print(f"   必要な環境変数:")
    print(f"   - MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DATABASE, MYSQL_PORT")
    print(f"   - MICROCMS_URL, MICROCMS_API_KEY")
    print(f"   - CONNPASS_URL, CONNPASS_NICKNAME")
    print(f"   - SITE_ROOT_URL")


class Config:
    """アプリケーション設定クラス"""
    
    # CORS設定
    ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com"
    ]
    
    # データベース設定
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "amenonuboko")
    MYSQL_PORT: str = os.getenv("MYSQL_PORT", "3306")
    
    # MicroCMS設定
    MICROCMS_URL: str = os.getenv("MICROCMS_URL", "")
    MICROCMS_API_KEY: str = os.getenv("MICROCMS_API_KEY", "")
    LIMIT: str = os.getenv("LIMIT", "100")
    
    # Connpass設定
    CONNPASS_URL: str = os.getenv("CONNPASS_URL", "https://connpass.com/api/v1/event/")
    CONNPASS_NICKNAME: str = os.getenv("CONNPASS_NICKNAME", "")
    
    # サイト設定
    SITE_ROOT_URL: str = os.getenv("SITE_ROOT_URL", "http://localhost:8000")


# グローバル設定インスタンス
config = Config()

def validate_config():
    """設定の検証を行う"""
    required_vars = [
        "MYSQL_USER",
        "MYSQL_HOST", 
        "MYSQL_DATABASE",
        "MICROCMS_URL",
        "MICROCMS_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(config, var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  以下の環境変数が設定されていません: {', '.join(missing_vars)}")
        print("   .envファイルを確認してください。")
        return False
    
    print("✅ 環境変数の設定が完了しました")
    return True

# アプリケーション起動時に設定を検証
if __name__ == "__main__":
    validate_config() 