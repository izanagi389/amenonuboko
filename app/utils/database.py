"""
データベース管理モジュール

このモジュールは、MySQLデータベースとの接続と操作を管理する
DatabaseManagerクラスを提供します。
"""

import logging
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple, Any

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

import config

# 設定値
MYSQL_CONFIG = {
    'user': config.MYSQL_USER,
    'password': config.MYSQL_PASSWORD, 
    'host': config.MYSQL_HOST,
    'database': config.MYSQL_DATABASE
}


class DatabaseManager:
    """
    データベース接続を管理するシングルトンクラス
    
    MySQLデータベースとの接続を管理し、効率的なデータベース操作を提供します。
    """
    
    _instance = None
    _engine = None
    _SessionLocal = None
    
    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """DatabaseManagerクラスの初期化"""
        if self._engine is None:
            self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """
        データベースエンジンを初期化
        
        接続プールとセッションファクトリを設定します。
        """
        database_url = 'mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4'.format(
            **MYSQL_CONFIG
        )
        
        self._engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False  # パフォーマンス向上のためSQLログを無効化
        )
        
        self._SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self._engine
        )
    
    @property
    def engine(self):
        """データベースエンジンを取得"""
        return self._engine
    
    @property
    def SessionLocal(self):
        """セッションファクトリを取得"""
        return self._SessionLocal
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        セッションコンテキストマネージャー
        
        Yields:
            SQLAlchemyセッション
            
        Raises:
            Exception: セッション操作中にエラーが発生した場合
        """
        session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """
        クエリを実行して結果を返す
        
        Args:
            query: SQLクエリ文字列
            params: クエリパラメータ
            
        Returns:
            クエリ結果のタプルリスト
            
        Raises:
            Exception: クエリ実行中にエラーが発生した場合
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            logging.error(f"クエリ実行エラー: {e}")
            raise
    
    def execute_query_dict(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        クエリを実行して辞書形式で結果を返す
        
        Args:
            query: SQLクエリ文字列
            params: クエリパラメータ
            
        Returns:
            クエリ結果の辞書リスト
            
        Raises:
            Exception: クエリ実行中にエラーが発生した場合
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            logging.error(f"クエリ実行エラー: {e}")
            raise
    
    def execute_query_numpy(self, query: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        クエリを実行してNumPy配列で結果を返す
        
        Args:
            query: SQLクエリ文字列
            params: クエリパラメータ
            
        Returns:
            クエリ結果のNumPy配列
            
        Raises:
            Exception: クエリ実行中にエラーが発生した場合
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                rows = result.fetchall()
                if not rows:
                    return np.array([])
                
                # NumPy配列に変換
                return np.array(rows)
        except Exception as e:
            logging.error(f"クエリ実行エラー: {e}")
            raise
    
    def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> int:
        """
        複数のパラメータでクエリを一括実行
        
        Args:
            query: SQLクエリ文字列
            params_list: パラメータのリスト
            
        Returns:
            実行された件数
            
        Raises:
            Exception: 一括実行中にエラーが発生した場合
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params_list)
                session.commit()
                return len(params_list)
        except Exception as e:
            logging.error(f"一括実行エラー: {e}")
            raise
    
    def insert_batch(self, table_name: str, data: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        バッチ挿入（高速版）
        
        Args:
            table_name: テーブル名
            data: 挿入するデータのリスト
            batch_size: バッチサイズ
            
        Returns:
            挿入された件数
            
        Raises:
            Exception: バッチ挿入中にエラーが発生した場合
        """
        if not data:
            return 0
        
        try:
            # カラム名を取得
            columns = list(data[0].keys())
            placeholders = ', '.join([':' + col for col in columns])
            
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                {', '.join([f"{col} = VALUES({col})" for col in columns if col != 'id'])}
            """
            
            total_inserted = 0
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                inserted = self.execute_many(query, batch)
                total_inserted += inserted
                
                if i % (batch_size * 10) == 0:
                    logging.info(f"バッチ挿入進捗: {i}/{len(data)} 件")
            
            logging.info(f"バッチ挿入完了: {total_inserted} 件")
            return total_inserted
            
        except Exception as e:
            logging.error(f"バッチ挿入エラー: {e}")
            raise
    
    def bulk_insert(self, table_name: str, columns: List[str], values: List[List[Any]], 
                   batch_size: int = 1000) -> int:
        """
        値リストを直接挿入（最高速版）
        
        Args:
            table_name: テーブル名
            columns: カラム名のリスト
            values: 値のリスト
            batch_size: バッチサイズ
            
        Returns:
            挿入された件数
            
        Raises:
            Exception: 一括挿入中にエラーが発生した場合
        """
        if not values:
            return 0
        
        try:
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({placeholders})
            """
            
            total_inserted = 0
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                
                with self.get_session() as session:
                    session.execute(text(query), batch)
                    session.commit()
                    total_inserted += len(batch)
                
                if i % (batch_size * 10) == 0:
                    logging.info(f"一括挿入進捗: {i}/{len(values)} 件")
            
            logging.info(f"一括挿入完了: {total_inserted} 件")
            return total_inserted
            
        except Exception as e:
            logging.error(f"一括挿入エラー: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        テーブルの存在確認
        
        Args:
            table_name: テーブル名
            
        Returns:
            テーブルが存在する場合True、そうでなければFalse
        """
        try:
            query = """
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = :database 
                AND table_name = :table_name
            """
            result = self.execute_query(query, {
                'database': MYSQL_CONFIG['database'],
                'table_name': table_name
            })
            return result[0][0] > 0
        except Exception as e:
            logging.error(f"テーブル存在確認エラー: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        テーブル情報を取得
        
        Args:
            table_name: テーブル名
            
        Returns:
            テーブル情報の辞書
        """
        try:
            query = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM information_schema.columns 
                WHERE table_schema = :database 
                AND table_name = :table_name
                ORDER BY ORDINAL_POSITION
            """
            result = self.execute_query_dict(query, {
                'database': MYSQL_CONFIG['database'],
                'table_name': table_name
            })
            return {'columns': result}
        except Exception as e:
            logging.error(f"テーブル情報取得エラー: {e}")
            return {'columns': []}
    
    def count_rows(self, table_name: str, where_clause: Optional[str] = None) -> int:
        """
        行数をカウント
        
        Args:
            table_name: テーブル名
            where_clause: WHERE句（オプション）
            
        Returns:
            行数
        """
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            result = self.execute_query(query)
            return result[0][0] if result else 0
        except Exception as e:
            logging.error(f"行数カウントエラー: {e}")
            return 0
    
    def clear_table(self, table_name: str) -> bool:
        """
        テーブルをクリア
        
        Args:
            table_name: テーブル名
            
        Returns:
            クリアが成功した場合True、失敗した場合False
        """
        try:
            # TRUNCATEの方が高速
            query = f"TRUNCATE TABLE {table_name}"
            with self.get_session() as session:
                session.execute(text(query))
                session.commit()
            return True
        except Exception:
            try:
                # TRUNCATEが失敗した場合はDELETEを使用
                query = f"DELETE FROM {table_name}"
                with self.get_session() as session:
                    session.execute(text(query))
                    session.commit()
                return True
            except Exception as e:
                logging.error(f"テーブルクリアエラー: {e}")
                return False
    
    def create_index(self, table_name: str, index_name: str, columns: List[str]) -> bool:
        """
        インデックスを作成
        
        Args:
            table_name: テーブル名
            index_name: インデックス名
            columns: インデックス対象のカラムリスト
            
        Returns:
            インデックス作成が成功した場合True、失敗した場合False
        """
        try:
            columns_str = ', '.join(columns)
            query = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
            with self.get_session() as session:
                session.execute(text(query))
                session.commit()
            return True
        except Exception as e:
            logging.error(f"インデックス作成エラー: {e}")
            return False
    
    def optimize_table(self, table_name: str) -> bool:
        """
        テーブルを最適化
        
        Args:
            table_name: テーブル名
            
        Returns:
            最適化が成功した場合True、失敗した場合False
        """
        try:
            query = f"OPTIMIZE TABLE {table_name}"
            with self.get_session() as session:
                session.execute(text(query))
                session.commit()
            return True
        except Exception as e:
            logging.error(f"テーブル最適化エラー: {e}")
            return False


# グローバルインスタンス
db_manager = DatabaseManager() 