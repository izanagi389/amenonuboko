# amenonuboko - 関連ブログタイトル抽出API

FastAPIとMySQLを使用した関連ブログタイトル抽出APIです。SentenceBERTを使用してブログタイトルの類似度を計算し、関連記事を推薦します。

## 🚀 機能

- **関連記事推薦**: SentenceBERTを使用した高精度な類似度計算
- **高速処理**: 並列処理とメモリ最適化による高速化
- **RESTful API**: FastAPIによる高性能なAPI
- **データベース**: MySQLによる永続化
- **Docker対応**: 完全なコンテナ化環境

## 📋 必要条件

- Docker
- Docker Compose
- 4GB以上のメモリ（推奨）

## 🛠️ セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd amenonuboko
```

### 2. 環境変数の設定

```bash
cp env.example .env
```

`.env`ファイルを編集して、必要な設定を行ってください：

```bash
# データベース設定
MYSQL_USER=amenonuboko_user
MYSQL_PASSWORD=amenonuboko_password
MYSQL_HOST=db
MYSQL_DATABASE=amenonuboko

# MicroCMS設定
MICROCMS_URL=https://your-microcms.microcms.io/api/v1/contents
MICROCMS_API_KEY=your_microcms_api_key

# Connpass設定
CONNPASS_URL=https://connpass.com/api/v1/event/
CONNPASS_NICKNAME=your_connpass_nickname
```

### 3. アプリケーションの起動

#### 基本起動
```bash
./scripts/start.sh
```

#### 開発環境（phpMyAdmin付き）
```bash
./scripts/start.sh dev
```

#### 本番環境（Nginx付き）
```bash
./scripts/start.sh prod
```

#### キャッシュ付き（Redis付き）
```bash
./scripts/start.sh cache
```

## 📱 アクセス情報

起動後、以下のURLでアクセスできます：

- **API**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs
- **ヘルスチェック**: http://localhost:8000/amenonuboko/health
- **phpMyAdmin** (開発環境): http://localhost:8080

## 🔧 API エンドポイント

### メインエンドポイント
```
GET /amenonuboko/
```

### 関連タイトル取得
```
GET /amenonuboko/v2/related_title/{blog_id}
```

### タグデータ取得
```
GET /amenonuboko/v1/tags/{blog_id}
```

### イベントデータ取得
```
GET /amenonuboko/v1/myEvents/{year_month}
```

### ヘルスチェック
```
GET /amenonuboko/health
```

## 🗄️ データベース

### 接続情報
- **ホスト**: localhost
- **ポート**: 3306
- **データベース**: amenonuboko
- **ユーザー**: amenonuboko_user
- **パスワード**: amenonuboko_password

### テーブル構造

#### related_data_v2
関連記事の類似度データを格納

```sql
CREATE TABLE related_data_v2 (
    id VARCHAR(255) NOT NULL,
    relate_id VARCHAR(255) NOT NULL,
    relate_title TEXT NOT NULL,
    bert_cos_distance FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id, relate_id)
);
```

#### topic_corpus
トピックコーパスデータを格納

```sql
CREATE TABLE topic_corpus (
    id VARCHAR(255) NOT NULL,
    corpus TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);
```

## 🔧 管理コマンド

### ログ確認
```bash
docker-compose logs -f
```

### アプリケーション再起動
```bash
docker-compose restart app
```

### データベース接続
```bash
docker-compose exec db mysql -u amenonuboko_user -p amenonuboko
```

### 全サービス停止
```bash
docker-compose down
```

### データベースリセット
```bash
docker-compose down -v
docker-compose up -d
```

## 🚀 パフォーマンス最適化

このアプリケーションは以下の最適化が実装されています：

- **並列処理**: マルチスレッド・マルチプロセスによる高速化
- **メモリ最適化**: 効率的なメモリ使用とガベージコレクション
- **データベース最適化**: インデックスとバッチ処理
- **ストリーミング処理**: 大量データの段階的処理

## 🔒 セキュリティ

- **SQLインジェクション対策**: パラメータ化クエリ
- **CORS設定**: 適切なクロスオリジン設定
- **セキュリティヘッダー**: XSS、CSRF対策
- **環境変数**: 機密情報の外部化

## 📊 監視・ログ

- **ヘルスチェック**: 自動的なサービス監視
- **ログ出力**: 構造化されたログ
- **メトリクス**: パフォーマンス監視
- **エラーハンドリング**: 包括的な例外処理

## 🤝 開発

### 開発環境のセットアップ

```bash
# 開発環境を起動
./scripts/start.sh dev

# ログを確認
docker-compose logs -f app

# コードを変更（ホットリロード対応）
# ファイルを編集すると自動的に再起動されます
```

### テスト

```bash
# テストの実行
docker-compose exec app python -m pytest

# カバレッジの確認
docker-compose exec app python -m pytest --cov=app
```

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🆘 トラブルシューティング

### よくある問題

1. **ポートが既に使用されている**
   ```bash
   # 使用中のポートを確認
   lsof -i :8000
   lsof -i :3306
   ```

2. **メモリ不足**
   ```bash
   # Dockerのメモリ制限を確認
   docker system df
   ```

3. **データベース接続エラー**
   ```bash
   # データベースのログを確認
   docker-compose logs db
   ```

### サポート

問題が発生した場合は、以下の手順で調査してください：

1. ログの確認: `docker-compose logs -f`
2. ヘルスチェック: `curl http://localhost:8000/amenonuboko/health`
3. データベース接続テスト: `docker-compose exec db mysql -u amenonuboko_user -p amenonuboko`

