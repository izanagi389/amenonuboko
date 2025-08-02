# amenonuboko API

FastAPIを使用した記事関連性計算API

## セットアップ

### 1. 環境変数の設定

```bash
# .envファイルを作成
cp env.example .env

# 必要に応じて値を編集
vim .env
```

### 2. データベース初期化

```bash
# テンプレートファイルをコピー
cp mysql/init/01-init.sql.template mysql/init/01-init.sql

# 環境変数に合わせて編集
vim mysql/init/01-init.sql
```

### 3. Docker Composeで起動

```bash
# 開発環境で起動
./scripts/start.sh dev

# または直接実行
docker compose up -d
```

## セキュリティ

- `mysql/init/01-init.sql`は機密情報を含むため、Gitから除外されています
- 本番環境では、適切なパスワードとユーザー名を設定してください
- テンプレートファイル（`01-init.sql.template`）を参考に設定してください

## API エンドポイント

- `GET /amenonuboko/health` - ヘルスチェック
- `GET /amenonuboko/v2/related_title/{id}` - 関連記事スコア取得

## 技術スタック

- FastAPI
- MySQL
- Docker
- 多言語SentenceBERT（記事関連性計算）

