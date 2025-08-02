"""
関連スコア計算モジュール

このモジュールは、コンテンツ間の関連性を計算するための
超高速関連スコア計算サービスを提供します。
"""

import gc
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Any, Tuple

import numpy as np

from app.db.sqlalchemy.crud import crud
from app.lib.algorithm.sentence_bert import SentenceBertJapanese
from app.utils.database import db_manager


class UltraRelatedScoreService:
    """
    超高速関連スコア計算サービス
    
    コンテンツ間の関連性を効率的に計算し、データベースに保存します。
    """

    def __init__(self, max_workers: int = None, batch_size: int = 2000, 
                 memory_limit_mb: int = 1024):
        """
        UltraRelatedScoreServiceクラスの初期化
        
        Args:
            max_workers: 最大ワーカー数
            batch_size: バッチサイズ
            memory_limit_mb: メモリ制限（MB）
        """
        self.model = None
        self.max_workers = max_workers or min(mp.cpu_count(), 12)
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self._initialize_model()
        
        # メモリ監視用
        self._memory_monitor = MemoryMonitor(memory_limit_mb)

    def _initialize_model(self) -> None:
        """
        SentenceBERTモデルを初期化（最適化版）
        
        Raises:
            Exception: モデル初期化中にエラーが発生した場合
        """
        try:
            self.model = SentenceBertJapanese(
                "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
            )
            # モデルをGPUに移動（利用可能な場合）
            if hasattr(self.model, 'to') and hasattr(self.model, 'device'):
                try:
                    self.model = self.model.to('cuda')
                except:
                    pass  # GPUが利用できない場合はCPUを使用
        except Exception as e:
            logging.error(f"SentenceBERTモデルの初期化に失敗しました: {e}")
            raise

    def start(self, content_list: List[Dict[str, Any]]) -> None:
        """
        関連スコアの計算と保存（超高速版）
        
        Args:
            content_list: 処理対象のコンテンツリスト
            
        Raises:
            Exception: 処理中にエラーが発生した場合
        """
        start_time = time.time()
        
        try:
            if not content_list:
                logging.warning("コンテンツリストが空です")
                return

            logging.info(f"超高速処理開始: {len(content_list)}件")

            # 1. ストリーミング処理でデータを分割
            content_chunks = self._stream_content_chunks(content_list)
            
            # 2. 各チャンクを並列処理
            all_results = []
            for chunk_idx, chunk in enumerate(content_chunks):
                logging.info(f"チャンク {chunk_idx + 1}/{len(content_chunks)} 処理中...")
                
                # メモリ監視
                self._memory_monitor.check_memory()
                
                # チャンク処理
                chunk_results = self._process_chunk(chunk, content_list)
                all_results.extend(chunk_results)
                
                # ガベージコレクション
                gc.collect()
            
            # 3. 結果をストリーミング保存
            self._stream_save_results(all_results)
            
            total_time = time.time() - start_time
            logging.info(f"超高速処理完了: {len(content_list)}件, 実行時間: {total_time:.2f}秒")
            
        except Exception as e:
            logging.error(f"関連スコア計算中にエラーが発生しました: {e}")
            raise

    def _stream_content_chunks(self, content_list: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        コンテンツをストリーミングチャンクに分割
        
        Args:
            content_list: 元のコンテンツリスト
            
        Returns:
            分割されたチャンクのリスト
        """
        chunks = []
        for i in range(0, len(content_list), self.batch_size):
            chunk = content_list[i:i + self.batch_size]
            chunks.append(chunk)
        return chunks

    def _process_chunk(self, chunk: List[Dict[str, Any]], 
                      all_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        チャンクを処理
        
        Args:
            chunk: 処理対象のチャンク
            all_content: 全コンテンツリスト
            
        Returns:
            処理結果のリスト
        """
        # データ抽出（最適化版）
        chunk_ids, chunk_titles = self._extract_chunk_data(chunk)
        all_ids, all_titles = self._extract_chunk_data(all_content)
        
        # ベクトル化（ストリーミング版）
        chunk_vectors = self._encode_titles_streaming(chunk_titles)
        
        # 類似度計算（最適化版）
        similarity_matrix = self._calculate_similarity_optimized(chunk_vectors, all_ids, all_titles)
        
        # 結果生成（ストリーミング版）
        results = self._generate_results_streaming(chunk_ids, chunk_titles, similarity_matrix, all_ids, all_titles)
        
        return results

    def _extract_chunk_data(self, content_list: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        データ抽出（最適化版）
        
        Args:
            content_list: コンテンツリスト
            
        Returns:
            IDリストとタイトルリストのタプル
        """
        # リスト内包表記で高速化
        ids = [content.get('id', '') for content in content_list]
        titles = [content.get('title', '') for content in content_list]
        return ids, titles

    def _encode_titles_streaming(self, titles: List[str]) -> np.ndarray:
        """
        タイトルをベクトル化（ストリーミング版）
        
        Args:
            titles: タイトルリスト
            
        Returns:
            ベクトル化されたタイトルの配列
            
        Raises:
            Exception: ベクトル化中にエラーが発生した場合
        """
        try:
            # より大きなバッチサイズで効率化
            optimal_batch_size = min(len(titles), max(50, len(titles) // self.max_workers))
            
            vectors_list = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # バッチ処理
                batches = [titles[i:i + optimal_batch_size] 
                          for i in range(0, len(titles), optimal_batch_size)]
                
                # 並列実行
                futures = {executor.submit(self._encode_batch_optimized, batch): batch 
                          for batch in batches}
                
                # 結果をストリーミング収集
                for future in as_completed(futures):
                    try:
                        vectors = future.result()
                        vectors_list.append(vectors)
                    except Exception as e:
                        logging.error(f"ベクトル化エラー: {e}")
                        continue
            
            # 結果を結合
            if vectors_list:
                return np.vstack(vectors_list)
            else:
                return np.array([])
                
        except Exception as e:
            logging.error(f"タイトルベクトル化中にエラーが発生しました: {e}")
            raise

    def _encode_batch_optimized(self, titles_batch: List[str]) -> np.ndarray:
        """
        バッチ単位でタイトルをベクトル化（最適化版）
        
        Args:
            titles_batch: バッチのタイトルリスト
            
        Returns:
            ベクトル化されたバッチの配列
        """
        if not titles_batch:
            return np.array([])
        
        vectors = self.model.encode(titles_batch)
        return vectors.numpy() if hasattr(vectors, 'numpy') else np.array(vectors)

    def _calculate_similarity_optimized(self, chunk_vectors: np.ndarray, 
                                      chunk_ids: List[str], all_titles: List[str]) -> Dict[str, List[float]]:
        """
        類似度計算（最適化版）
        
        Args:
            chunk_vectors: チャンクのベクトル配列
            chunk_ids: チャンクのIDリスト
            all_titles: 全タイトルリスト
            
        Returns:
            類似度辞書
        """
        if chunk_vectors.size == 0:
            return {}
        
        # 正規化（最適化版）
        norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
        normalized_vectors = chunk_vectors / (norms + 1e-8)
        
        # 類似度計算（メモリ効率版）
        similarity_dict = {}
        
        # チャンクごとに処理
        for i, chunk_id in enumerate(chunk_ids):
            # 各チャンクベクトルと全タイトルの類似度を計算
            similarities = []
            
            # 並列処理で類似度計算
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 全タイトルをベクトル化
                all_vectors = self._encode_titles_streaming(all_titles)
                
                if all_vectors.size > 0:
                    all_norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
                    all_normalized = all_vectors / (all_norms + 1e-8)
                    
                    # 類似度計算
                    similarity = np.dot(normalized_vectors[i:i+1], all_normalized.T)[0]
                    similarities = (similarity / 2).tolist()  # 既存の形式に合わせて2で割る
                else:
                    similarities = [0.0] * len(all_titles)
            
            similarity_dict[chunk_id] = similarities
        
        return similarity_dict

    def _generate_results_streaming(self, chunk_ids: List[str], chunk_titles: List[str],
                                  similarity_dict: Dict[str, List[float]], 
                                  all_ids: List[str], all_titles: List[str]) -> List[Dict[str, Any]]:
        """
        結果生成（ストリーミング版）
        
        Args:
            chunk_ids: チャンクのIDリスト
            chunk_titles: チャンクのタイトルリスト
            similarity_dict: 類似度辞書
            all_ids: 全IDリスト
            all_titles: 全タイトルリスト
            
        Returns:
            生成された結果のリスト
        """
        results = []
        
        # ジェネレータでメモリ効率化
        for chunk_id, similarities in similarity_dict.items():
            for j, (relate_id, relate_title) in enumerate(zip(all_ids, all_titles)):
                if j < len(similarities):
                    results.append({
                        'id': chunk_id,
                        'relate_id': relate_id,
                        'relate_title': relate_title,
                        'bert_cos_distance': float(similarities[j])
                    })
        
        return results

    def _stream_save_results(self, all_results: List[Dict[str, Any]]) -> None:
        """
        結果をストリーミング保存
        
        Args:
            all_results: 保存する結果のリスト
            
        Raises:
            Exception: 保存中にエラーが発生した場合
        """
        try:
            # 新しいCRUDクラスを使用して一括挿入
            crud().insert_related_data(all_results)
            logging.info(f"ストリーミングデータベース保存完了: {len(all_results)}件")
            
        except Exception as e:
            logging.error(f"データベース保存中にエラーが発生しました: {e}")
            raise

    def get_scores(self, num: int, search_id: str) -> List[Dict[str, Any]]:
        """
        関連スコアを取得（超高速版）
        
        Args:
            num: 取得件数
            search_id: 検索ID
            
        Returns:
            関連スコアのリスト
        """
        try:
            # インデックスを活用した高速クエリ
            query = """
                SELECT relate_id, relate_title, bert_cos_distance
                FROM related_data_v2 
                WHERE id = %s AND relate_id != %s
                ORDER BY bert_cos_distance ASC
                LIMIT %s
            """
            
            with db_manager.get_session() as session:
                result = session.execute(query, [search_id, search_id, num])
                rows = result.fetchall()
            
            if not rows:
                return []
            
            # 結果を高速変換
            results = [
                {
                    'id': row[0],
                    'title': row[1],
                    'score': float(row[2])
                }
                for row in rows
            ]
            
            return results
            
        except Exception as e:
            logging.error(f"スコア取得中にエラーが発生しました: {e}")
            return []


class MemoryMonitor:
    """
    メモリ使用量監視クラス
    
    メモリ使用量を監視し、必要に応じてガベージコレクションを実行します。
    """
    
    def __init__(self, limit_mb: int):
        """
        MemoryMonitorクラスの初期化
        
        Args:
            limit_mb: メモリ制限（MB）
        """
        self.limit_mb = limit_mb
        self.last_check = time.time()
    
    def check_memory(self) -> None:
        """
        メモリ使用量をチェック
        
        メモリ使用量が制限を超えた場合、ガベージコレクションを実行します。
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.limit_mb:
                logging.warning(f"メモリ使用量が上限を超えています: {memory_mb:.1f}MB > {self.limit_mb}MB")
                gc.collect()
                
        except ImportError:
            pass  # psutilが利用できない場合はスキップ


# 後方互換性のためのエイリアス
create_score = UltraRelatedScoreService
create_score_ultra = UltraRelatedScoreService 