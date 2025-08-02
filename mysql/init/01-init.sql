-- amenonuboko データベースの初期化スクリプト

-- データベースの作成（既に存在する場合はスキップ）
CREATE DATABASE IF NOT EXISTS amenonuboko CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- データベースの使用
USE amenonuboko;

-- ユーザーの作成と権限付与
CREATE USER IF NOT EXISTS 'amenonuboko_user'@'%' IDENTIFIED BY 'amenonuboko_password';
GRANT ALL PRIVILEGES ON amenonuboko.* TO 'amenonuboko_user'@'%';
FLUSH PRIVILEGES;

-- 関連データテーブルの作成
CREATE TABLE IF NOT EXISTS related_data_v2 (
    id VARCHAR(255) NOT NULL,
    relate_id VARCHAR(255) NOT NULL,
    relate_title TEXT NOT NULL,
    bert_cos_distance FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id, relate_id),
    INDEX idx_id (id),
    INDEX idx_relate_id (relate_id),
    INDEX idx_distance (bert_cos_distance)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- トピックコーパステーブルの作成
CREATE TABLE IF NOT EXISTS topic_corpus (
    id VARCHAR(255) NOT NULL,
    corpus TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- コンテンツテーブルの作成（オプション）
CREATE TABLE IF NOT EXISTS contents (
    id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    blog_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_title (title(100)),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ログテーブルの作成（オプション）
CREATE TABLE IF NOT EXISTS api_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT NOT NULL,
    response_time FLOAT NOT NULL,
    user_agent TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_endpoint (endpoint),
    INDEX idx_status_code (status_code),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- サンプルデータの挿入（テスト用）
INSERT IGNORE INTO related_data_v2 (id, relate_id, relate_title, bert_cos_distance) VALUES
('test_0001', 'test_0002', 'Pythonで機械学習を始める方法 - Part 2', 0.85),
('test_0001', 'test_0003', 'FastAPIを使ったWebアプリケーション開発 - Part 3', 0.72),
('test_0002', 'test_0001', 'Pythonで機械学習を始める方法 - Part 1', 0.85),
('test_0002', 'test_0003', 'FastAPIを使ったWebアプリケーション開発 - Part 3', 0.68);

INSERT IGNORE INTO topic_corpus (id, corpus) VALUES
('test_0001', 'Python機械学習 データ分析 アルゴリズム'),
('test_0002', 'Python機械学習 深層学習 ニューラルネットワーク');

-- 権限の確認
SHOW GRANTS FOR 'amenonuboko_user'@'%'; 