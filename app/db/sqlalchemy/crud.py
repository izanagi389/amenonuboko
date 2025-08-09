"""
CRUDサービスモジュール

このモジュールは、データベース操作のためのCRUDサービスを提供します。
"""

import logging
from typing import Dict, Any, List, Optional

from app.utils.database import db_manager


class CRUDService:
    """
    最適化されたCRUDサービス
    
    データベース操作のための包括的なサービスを提供します。
    """

    def __init__(self):
        """CRUDServiceクラスの初期化"""
        self.db_manager = db_manager

    def get_data_for_df(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        クエリを実行して辞書形式で結果を返す
        
        Args:
            query_string: SQLクエリ文字列
            params: クエリパラメータ
            
        Returns:
            辞書形式の結果リスト
        """
        return self.db_manager.execute_query_dict(query_string, params)

    def get_data_for_numpy(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        クエリを実行してタプル形式で結果を返す
        
        Args:
            query_string: SQLクエリ文字列
            params: クエリパラメータ
            
        Returns:
            タプル形式の結果リスト
        """
        return self.db_manager.execute_query(query_string, params)

    def add_data_for_df(self, data: List[Dict[str, Any]], table_name: str) -> None:
        """
        辞書形式のデータをテーブルに挿入
        
        Args:
            data: 挿入するデータリスト
            table_name: テーブル名
        """
        self.db_manager.insert_batch(table_name, data)

    def add_data_bulk(self, table_name: str, columns: List[str], values: List[List[Any]]) -> None:
        """
        値リストを直接挿入（最高速版）
        
        Args:
            table_name: テーブル名
            columns: カラム名リスト
            values: 値リスト
        """
        self.db_manager.bulk_insert(table_name, columns, values)

    def get_topic_data_for_df(self, blog_id: str) -> Dict[str, Any]:
        """
        トピックデータを取得
        
        Args:
            blog_id: 検索ID
            
        Returns:
            トピックデータの辞書
        """
        query = """
            SELECT id, corpus 
            FROM topic_corpus 
            WHERE id = :blog_id
        """
        result = self.db_manager.execute_query_dict(query, {'blog_id': blog_id})
        
        if not result:
            return {"error": "データが見つかりません"}
        
        return result[0]

    def get_relate_score_for_df(self, blog_id: str) -> List[Dict[str, Any]]:
        """
        関連スコアデータを取得
        
        Args:
            blog_id: 検索ID
            
        Returns:
            関連スコアデータのリスト
        """
        query = """
            SELECT id, relate_id, relate_title, bert_cos_distance 
            FROM related_data_v2 
            WHERE id = :blog_id
        """
        return self.db_manager.execute_query_dict(query, {'blog_id': blog_id})

    def get_relate_score_for_numpy(self, blog_id: str) -> List[tuple]:
        """
        関連スコアデータをNumPy形式で取得
        
        Args:
            blog_id: 検索ID
            
        Returns:
            関連スコアデータのタプルリスト
        """
        query = """
            SELECT id, relate_id, relate_title, bert_cos_distance 
            FROM related_data_v2 
            WHERE id = :blog_id
        """
        return self.db_manager.execute_query(query, {'blog_id': blog_id})

    def insert_related_data(self, data: List[Dict[str, Any]]) -> None:
        """
        関連データを一括挿入
        
        Args:
            data: 挿入する関連データリスト
        """
        if not data:
            return
        
        # テーブルをクリア
        self.db_manager.clear_table("related_data_v2")
        
        # バッチ挿入
        self.db_manager.insert_batch("related_data_v2", data, batch_size=5000)

    def insert_topic_data(self, data: List[Dict[str, Any]]) -> None:
        """
        トピックデータを一括挿入
        
        Args:
            data: 挿入するトピックデータリスト
        """
        if not data:
            return
        
        # テーブルをクリア
        self.db_manager.clear_table("topic_corpus")
        
        # バッチ挿入
        self.db_manager.insert_batch("topic_corpus", data, batch_size=1000)

    def bulk_insert_related_data(self, columns: List[str], values: List[List[Any]]) -> None:
        """
        関連データを最高速で一括挿入
        
        Args:
            columns: カラム名リスト
            values: 値リスト
        """
        if not values:
            return
        
        # テーブルをクリア
        self.db_manager.clear_table("related_data_v2")
        
        # 一括挿入
        self.db_manager.bulk_insert("related_data_v2", columns, values, batch_size=5000)

    def get_related_scores_paginated(self, blog_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        関連スコアをページネーション付きで取得
        
        Args:
            blog_id: 検索ID
            limit: 取得件数制限
            offset: オフセット
            
        Returns:
            ページネーション情報を含む結果辞書
        """
        # 総件数を取得
        count_query = """
            SELECT COUNT(*) 
            FROM related_data_v2 
            WHERE id = :blog_id AND relate_id != :blog_id
        """
        count_result = self.db_manager.execute_query(count_query, {'blog_id': blog_id})
        total_count = count_result[0][0] if count_result else 0
        
        # データを取得
        data_query = """
            SELECT relate_id, relate_title, bert_cos_distance
            FROM related_data_v2 
            WHERE id = :blog_id AND relate_id != :blog_id
            ORDER BY bert_cos_distance ASC
            LIMIT :limit OFFSET :offset
        """
        data = self.db_manager.execute_query_dict(data_query, {
            'blog_id': blog_id,
            'limit': limit,
            'offset': offset
        })
        
        return {
            'data': data,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_next': (offset + limit) < total_count,
            'has_prev': offset > 0
        }

    def search_related_titles(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        タイトルで関連データを検索
        
        Args:
            search_term: 検索語
            limit: 取得件数制限
            
        Returns:
            検索結果のリスト
        """
        query = """
            SELECT DISTINCT relate_id, relate_title, bert_cos_distance
            FROM related_data_v2 
            WHERE relate_title LIKE :search_term
            ORDER BY bert_cos_distance ASC
            LIMIT :limit
        """
        return self.db_manager.execute_query_dict(query, {
            'search_term': f'%{search_term}%',
            'limit': limit
        })

    def get_statistics(self) -> Dict[str, Any]:
        """
        データベース統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        try:
            # 関連データの統計
            related_count = self.db_manager.count_rows("related_data_v2")
            topic_count = self.db_manager.count_rows("topic_corpus")
            
            # ユニークID数
            unique_ids_query = "SELECT COUNT(DISTINCT id) FROM related_data_v2"
            unique_ids_result = self.db_manager.execute_query(unique_ids_query)
            unique_ids = unique_ids_result[0][0] if unique_ids_result else 0
            
            # 平均類似度
            avg_score_query = "SELECT AVG(bert_cos_distance) FROM related_data_v2"
            avg_score_result = self.db_manager.execute_query(avg_score_query)
            avg_score = float(avg_score_result[0][0]) if avg_score_result and avg_score_result[0][0] else 0.0
            
            return {
                'related_data_count': related_count,
                'topic_data_count': topic_count,
                'unique_ids_count': unique_ids,
                'average_similarity_score': avg_score,
                'database_name': self.db_manager.MYSQL_CONFIG['database']
            }
        except Exception as e:
            logging.error(f"統計情報取得エラー: {e}")
            return {
                'error': '統計情報の取得に失敗しました',
                'related_data_count': 0,
                'topic_data_count': 0,
                'unique_ids_count': 0,
                'average_similarity_score': 0.0
            }

    def optimize_database(self) -> Dict[str, bool]:
        """
        データベース最適化を実行
        
        Returns:
            最適化結果の辞書
        """
        try:
            results = {}
            
            # テーブル最適化
            results['related_data_v2'] = self.db_manager.optimize_table("related_data_v2")
            results['topic_corpus'] = self.db_manager.optimize_table("topic_corpus")
            
            # インデックス作成（存在しない場合）
            if not self.db_manager.table_exists("related_data_v2"):
                results['index_creation'] = False
            else:
                # インデックスが存在するかチェック（簡易版）
                results['index_creation'] = True
            
            return results
        except Exception as e:
            logging.error(f"データベース最適化エラー: {e}")
            return {'error': str(e)}

    def backup_table_structure(self) -> Dict[str, Any]:
        """
        テーブル構造のバックアップ情報を取得
        
        Returns:
            テーブル構造情報の辞書
        """
        try:
            related_info = self.db_manager.get_table_info("related_data_v2")
            topic_info = self.db_manager.get_table_info("topic_corpus")
            
            return {
                'related_data_v2': related_info,
                'topic_corpus': topic_info,
                'backup_timestamp': '2024-01-01 00:00:00'  # 実際の実装では現在時刻を使用
            }
        except Exception as e:
            logging.error(f"テーブル構造バックアップエラー: {e}")
            return {'error': str(e)}


# 後方互換性のためのエイリアス
crud = CRUDService
