#!/bin/bash

# amenonuboko アプリケーション起動スクリプト

set -e

echo "🚀 amenonuboko アプリケーションを起動します..."

# 環境変数ファイルの確認
if [ ! -f .env ]; then
    echo "⚠️  .env ファイルが見つかりません。env.example をコピーして設定してください。"
    echo "cp env.example .env"
    exit 1
fi

# Docker Compose の確認
if ! command -v docker compose &> /dev/null; then
    echo "❌ docker compose がインストールされていません。"
    exit 1
fi

# 開発環境の起動
if [ "$1" = "dev" ]; then
    echo "🔧 開発環境を起動します..."
    docker compose --profile dev up -d
    
    echo "📊 phpMyAdmin: http://localhost:8080"
    echo "   - ユーザー: amenonuboko_user"
    echo "   - パスワード: amenonuboko_password"
    
elif [ "$1" = "prod" ]; then
    echo "🏭 本番環境を起動します..."
    docker compose --profile production up -d
    
elif [ "$1" = "cache" ]; then
    echo "⚡ キャッシュ付きで起動します..."
    docker compose --profile cache up -d
    
else
    echo "🔧 基本環境を起動します..."
    docker compose up -d
fi

# ヘルスチェック
echo "⏳ サービス起動を待機中..."
sleep 10

# アプリケーションのヘルスチェック
echo "🔍 アプリケーションのヘルスチェック..."
for i in {1..30}; do
    if curl -f http://localhost:8000/amenonuboko/health > /dev/null 2>&1; then
        echo "✅ アプリケーションが正常に起動しました！"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ アプリケーションの起動に失敗しました。"
        echo "ログを確認してください: docker compose logs app"
        exit 1
    fi
    
    echo "⏳ 起動中... ($i/30)"
    sleep 2
done

# データベースのヘルスチェック
echo "🔍 データベースのヘルスチェック..."
for i in {1..30}; do
    if docker compose exec -T db mysqladmin ping -h localhost -u root -proot_password > /dev/null 2>&1; then
        echo "✅ データベースが正常に起動しました！"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ データベースの起動に失敗しました。"
        echo "ログを確認してください: docker compose logs db"
        exit 1
    fi
    
    echo "⏳ データベース起動中... ($i/30)"
    sleep 2
done

echo ""
echo "🎉 amenonuboko アプリケーションが正常に起動しました！"
echo ""
echo "📱 アクセス情報:"
echo "   - API: http://localhost:8000"
echo "   - ヘルスチェック: http://localhost:8000/amenonuboko/health"
echo "   - API ドキュメント: http://localhost:8000/docs"
echo ""
echo "🗄️  データベース情報:"
echo "   - ホスト: localhost"
echo "   - ポート: 3306"
echo "   - データベース: amenonuboko"
echo "   - ユーザー: amenonuboko_user"
echo "   - パスワード: amenonuboko_password"
echo ""
echo "🔧 管理コマンド:"
echo "   - ログ確認: docker compose logs -f"
echo "   - 停止: docker compose down"
echo "   - 再起動: docker compose restart"
echo "   - データベース接続: docker compose exec db mysql -u amenonuboko_user -p amenonuboko"
echo "" 