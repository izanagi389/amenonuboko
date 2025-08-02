"""
多言語SentenceBERTモジュール

このモジュールは、多言語対応のSentenceBERTモデルを使用して
記事の関連性を計算するための実装を提供します。
"""

import logging
import numpy as np
from typing import List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MultilingualSentenceBert:
    """
    多言語SentenceBERTクラス
    
    多言語対応のSentenceBERTモデルを使用して、記事の関連性を計算します。
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", 
                 device: Optional[str] = None, max_length: int = 512):
        """
        MultilingualSentenceBertクラスの初期化
        
        Args:
            model_name: 使用するモデル名
            device: 使用するデバイス（cuda/cpu）
            max_length: 最大トークン長
        """
        self.logger = logging.getLogger(__name__)
        self.max_length = max_length
        
        try:
            # デバイスの設定
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            
            # モデルの初期化
            self.logger.info(f"多言語SentenceBERTモデルを初期化中: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            
            # モデルの設定
            self.model.max_seq_length = max_length
            
            # モデルの動作確認
            try:
                test_text = "テスト"
                test_embedding = self.model.encode([test_text], convert_to_numpy=True, batch_size=1)
                if test_embedding is None or test_embedding.size == 0:
                    raise Exception("モデルのテスト実行に失敗しました")
                self.logger.debug(f"モデルテスト成功: 埋め込み次元数 {test_embedding.shape}")
            except Exception as test_e:
                self.logger.warning(f"モデルテストでエラー: {test_e}")
                # テストエラーでも続行（実際の処理で対応）
            
            self.logger.info(f"多言語SentenceBERTモデルの初期化完了: {model_name}")
            
        except Exception as e:
            self.logger.error(f"多言語SentenceBERTモデルの初期化に失敗しました: {e}")
            import traceback
            self.logger.error(f"詳細エラー: {traceback.format_exc()}")
            
            # フォールバック用の軽量モデル
            try:
                self.logger.info("フォールバック用の軽量モデルを試行中...")
                fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model = SentenceTransformer(fallback_model, device="cpu")
                self.model.max_seq_length = max_length
                
                # フォールバックモデルの動作確認
                try:
                    test_text = "テスト"
                    test_embedding = self.model.encode([test_text], convert_to_numpy=True, batch_size=1)
                    if test_embedding is None or test_embedding.size == 0:
                        raise Exception("フォールバックモデルのテスト実行に失敗しました")
                    self.logger.debug(f"フォールバックモデルテスト成功: 埋め込み次元数 {test_embedding.shape}")
                except Exception as test_e:
                    self.logger.warning(f"フォールバックモデルテストでエラー: {test_e}")
                    # テストエラーでも続行（実際の処理で対応）
                
                self.logger.info(f"フォールバックモデルの初期化完了: {fallback_model}")
            except Exception as fallback_e:
                self.logger.error(f"フォールバックモデルの初期化にも失敗しました: {fallback_e}")
                import traceback
                self.logger.error(f"フォールバック詳細エラー: {traceback.format_exc()}")
                raise Exception("SentenceBERTモデルの初期化に完全に失敗しました")
    
    def encode(self, texts: List[str], batch_size: int = 8, 
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        テキストをベクトル化
        
        Args:
            texts: ベクトル化するテキストのリスト
            batch_size: バッチサイズ
            normalize_embeddings: 埋め込みを正規化するかどうか
            
        Returns:
            ベクトル化されたテキストの配列
        """
        try:
            if not texts:
                self.logger.debug("テキストリストが空です")
                return np.array([])
            
            # モデルの状態確認
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("モデルが初期化されていません")
                return np.array([])
            
            # テキストの前処理
            processed_texts = self._preprocess_texts(texts)
            
            if not processed_texts:
                self.logger.warning("前処理後のテキストが空です")
                return np.array([])
            
            self.logger.debug(f"ベクトル化対象テキスト数: {len(processed_texts)}")
            
            # バッチサイズを小さくして安定性を向上
            safe_batch_size = min(batch_size, 2)  # さらに小さく
            
            # 個別処理でエラーを回避
            all_embeddings = []
            
            for i in range(0, len(processed_texts), safe_batch_size):
                batch_texts = processed_texts[i:i + safe_batch_size]
                
                try:
                    # バッチベクトル化
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=1,  # 1つずつ処理
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    if batch_embeddings is not None and batch_embeddings.size > 0:
                        all_embeddings.append(batch_embeddings)
                    else:
                        # エラーの場合はゼロベクトルを追加
                        self.logger.warning(f"バッチ {i//safe_batch_size + 1} で空の埋め込みが返されました")
                        zero_embedding = np.zeros((len(batch_texts), 384))  # 384次元
                        all_embeddings.append(zero_embedding)
                        
                except Exception as batch_e:
                    self.logger.error(f"バッチ {i//safe_batch_size + 1} でエラー: {batch_e}")
                    # エラーの場合はゼロベクトルを追加
                    zero_embedding = np.zeros((len(batch_texts), 384))  # 384次元
                    all_embeddings.append(zero_embedding)
            
            # 結果を結合
            if all_embeddings:
                return np.vstack(all_embeddings)
            else:
                self.logger.warning("すべてのバッチでエラーが発生しました")
                return np.array([])
            
        except Exception as e:
            self.logger.error(f"テキストベクトル化エラー: {e}")
            import traceback
            self.logger.error(f"詳細エラー: {traceback.format_exc()}")
            return np.array([])
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        2つのテキスト間の類似度を計算
        
        Args:
            text1, text2: 比較するテキスト
            
        Returns:
            類似度スコア（0.0-1.0）
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # ベクトル化
            embeddings = self.encode([text1, text2])
            
            if embeddings.size == 0:
                return 0.0
            
            # コサイン類似度を計算
            similarity = cosine_similarity(
                embeddings[0:1], 
                embeddings[1:2]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"類似度計算エラー: {e}")
            return 0.0
    
    def calculate_batch_similarity(self, texts: List[str]) -> np.ndarray:
        """
        複数のテキスト間の類似度行列を計算
        
        Args:
            texts: 比較するテキストのリスト
            
        Returns:
            類似度行列
        """
        try:
            if not texts:
                return np.array([])
            
            # ベクトル化
            embeddings = self.encode(texts)
            
            if embeddings.size == 0:
                return np.array([])
            
            # 類似度行列を計算
            similarity_matrix = cosine_similarity(embeddings)
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"バッチ類似度計算エラー: {e}")
            return np.array([])
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         top_k: int = 5) -> List[tuple]:
        """
        クエリテキストに最も類似したテキストを見つける
        
        Args:
            query_text: クエリテキスト
            candidate_texts: 候補テキストのリスト
            top_k: 返す類似テキストの数
            
        Returns:
            (インデックス, 類似度スコア) のタプルのリスト
        """
        try:
            if not query_text or not candidate_texts:
                return []
            
            # クエリと候補を結合してベクトル化
            all_texts = [query_text] + candidate_texts
            embeddings = self.encode(all_texts)
            
            if embeddings.size == 0:
                return []
            
            # クエリベクトル
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            # 類似度を計算
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # 上位k個を取得
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((idx, float(similarities[idx])))
            
            return results
            
        except Exception as e:
            self.logger.error(f"類似テキスト検索エラー: {e}")
            return []
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        テキストの前処理
        
        Args:
            texts: 前処理するテキストのリスト
            
        Returns:
            前処理されたテキストのリスト
        """
        processed_texts = []
        
        for text in texts:
            if text and isinstance(text, str):
                # 基本的な前処理
                processed_text = text.strip()
                
                # 空文字列でない場合のみ追加
                if processed_text:
                    processed_texts.append(processed_text)
                else:
                    self.logger.warning("空のテキストをスキップしました")
            else:
                self.logger.warning(f"無効なテキストをスキップしました: {type(text)}")
        
        return processed_texts
    
    def get_model_info(self) -> dict:
        """
        モデル情報を取得
        
        Returns:
            モデル情報の辞書
        """
        try:
            return {
                'model_name': self.model.get_sentence_embedding_dimension(),
                'max_length': self.max_length,
                'device': str(self.device),
                'embedding_dimension': self.model.get_sentence_embedding_dimension()
            }
        except Exception as e:
            self.logger.error(f"モデル情報取得エラー: {e}")
            return {}


# 便利関数
def create_multilingual_bert(model_name: str = "intfloat/multilingual-e5-large") -> MultilingualSentenceBert:
    """
    多言語SentenceBERTインスタンスを作成する便利関数
    
    Args:
        model_name: 使用するモデル名
        
    Returns:
        MultilingualSentenceBertインスタンス
    """
    return MultilingualSentenceBert(model_name=model_name)


def calculate_text_similarity(text1: str, text2: str, 
                            model_name: str = "intfloat/multilingual-e5-large") -> float:
    """
    2つのテキスト間の類似度を計算する便利関数
    
    Args:
        text1, text2: 比較するテキスト
        model_name: 使用するモデル名
        
    Returns:
        類似度スコア
    """
    try:
        bert = MultilingualSentenceBert(model_name=model_name)
        return bert.calculate_similarity(text1, text2)
    except Exception as e:
        logging.error(f"類似度計算エラー: {e}")
        return 0.0 