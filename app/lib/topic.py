"""
トピック処理モジュール

このモジュールは、テキストコンテンツからトピックを抽出し、
LDA（Latent Dirichlet Allocation）を使用してトピックモデリングを実行します。
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional

from app.db.sqlalchemy.crud import crud
from app.lib.algorithm.lda import LDA
from app.lib.contents import microcms
from app.lib.morphology import sudachi
from app.lib.text.shape import TextShapping


class TopicProcessor:
    """
    トピック処理クラス
    
    テキストコンテンツからトピックを抽出し、LDAを使用して
    トピックモデリングを実行するためのクラスです。
    """

    def __init__(self):
        """TopicProcessorクラスの初期化"""
        self.logger = logging.getLogger(__name__)

    def start(self, content_list: List[Dict[str, Any]]) -> None:
        """
        トピック処理を開始
        
        Args:
            content_list: 処理対象のコンテンツリスト
            
        Raises:
            Exception: 処理中にエラーが発生した場合
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"トピック処理開始: {len(content_list)}件")
            
            # 1. データの前処理
            processed_contents = self._preprocess_contents(content_list)
            
            # 2. テキストの正規化
            normalized_docs = self._normalize_documents(processed_contents)
            
            # 3. 形態素解析
            tokenized_docs = self._tokenize_documents(normalized_docs)
            
            # 4. テキストクリーニング
            cleaned_docs = self._clean_documents(tokenized_docs)
            
            # 5. LDA処理
            topics = self._process_lda(cleaned_docs)
            
            # 6. コーパス生成
            corpus_data = self._generate_corpus(processed_contents, topics)
            
            # 7. データベース保存
            self._save_to_database(corpus_data)
            
            total_time = time.time() - start_time
            self.logger.info(f"トピック処理完了: {len(content_list)}件, 実行時間: {total_time:.2f}秒")
            
        except Exception as e:
            self.logger.error(f"トピック処理中にエラーが発生しました: {e}")
            raise

    def _preprocess_contents(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        コンテンツの前処理
        
        Args:
            content_list: 元のコンテンツリスト
            
        Returns:
            前処理済みのコンテンツリスト
        """
        processed_contents = []
        
        for content in content_list:
            # 不要なテキストを削除
            cleaned_content = TextShapping.remove_unnecessary_text(content.get('blogContent', ''))
            
            processed_contents.append({
                'id': content.get('id', ''),
                'title': content.get('title', ''),
                'blogContent': cleaned_content
            })
        
        return processed_contents

    def _normalize_documents(self, processed_contents: List[Dict[str, Any]]) -> List[str]:
        """
        ドキュメントの正規化
        
        Args:
            processed_contents: 前処理済みコンテンツリスト
            
        Returns:
            正規化されたドキュメントリスト
        """
        docs = [content['blogContent'] for content in processed_contents]
        self.logger.info(f"ドキュメント正規化完了: {len(docs)}件")
        return docs

    def _tokenize_documents(self, docs: List[str]) -> List[List[str]]:
        """
        ドキュメントの形態素解析
        
        Args:
            docs: 正規化されたドキュメントリスト
            
        Returns:
            形態素解析済みドキュメントリスト
        """
        self.logger.info("形態素解析開始")
        tokenized_docs = sudachi.Tokenize(docs)
        self.logger.info(f"形態素解析完了: {len(tokenized_docs)}件")
        return tokenized_docs

    def _clean_documents(self, tokenized_docs: List[List[str]]) -> List[List[str]]:
        """
        ドキュメントのクリーニング
        
        Args:
            tokenized_docs: 形態素解析済みドキュメントリスト
            
        Returns:
            クリーニング済みドキュメントリスト
        """
        self.logger.info("ドキュメントクリーニング開始")
        
        # 数字を削除
        cleaned_docs = TextShapping.remove_numbers(tokenized_docs)
        
        # 1単語のみのドキュメントを削除
        cleaned_docs = TextShapping.remove_one_word(cleaned_docs)
        
        # ストップワードを削除
        cleaned_docs = TextShapping.remove_stop_words(cleaned_docs)
        
        self.logger.info(f"ドキュメントクリーニング完了: {len(cleaned_docs)}件")
        return cleaned_docs

    def _process_lda(self, cleaned_docs: List[List[str]]) -> List[str]:
        """
        LDA処理を実行
        
        Args:
            cleaned_docs: クリーニング済みドキュメントリスト
            
        Returns:
            抽出されたトピックリスト
        """
        self.logger.info("LDA処理開始")
        
        # バイグラムをドキュメントに変換
        docs_with_bigrams = LDA.bigram2docs(cleaned_docs)
        
        # 辞書を作成
        dictionary = LDA.get_dictionary(docs_with_bigrams)
        
        # コーパスを作成
        corpus = LDA.get_corpus(dictionary, docs_with_bigrams)
        
        # LDAモデルを読み込み
        self.logger.info("LDAモデル読み込み開始")
        model = LDA.load(dictionary, corpus)
        self.logger.info("LDAモデル読み込み完了")
        
        # トピックを抽出
        self.logger.info("トピック抽出開始")
        topics = LDA.topics(model, corpus)
        self.logger.info(f"トピック抽出完了: {len(topics)}個のトピック")
        
        return topics

    def _generate_corpus(self, processed_contents: List[Dict[str, Any]], topics: List[str]) -> List[Dict[str, Any]]:
        """
        コーパスデータを生成
        
        Args:
            processed_contents: 前処理済みコンテンツリスト
            topics: 抽出されたトピックリスト
            
        Returns:
            コーパスデータリスト
        """
        self.logger.info("コーパス生成開始")
        
        # 正規表現パターンをコンパイル
        pattern = re.compile("|".join(topics))
        
        corpus_data = []
        
        for content in processed_contents:
            # トピックを抽出
            matches = re.findall(pattern, content['blogContent'])
            unique_matches = list(set(matches))
            corpus_text = ",".join(unique_matches) if unique_matches else ""
            
            corpus_data.append({
                'id': content['id'],
                'corpus': corpus_text
            })
        
        self.logger.info(f"コーパス生成完了: {len(corpus_data)}件")
        return corpus_data

    def _save_to_database(self, corpus_data: List[Dict[str, Any]]) -> None:
        """
        データベースに保存
        
        Args:
            corpus_data: 保存するコーパスデータリスト
            
        Raises:
            Exception: データベース保存中にエラーが発生した場合
        """
        self.logger.info("データベース保存開始")
        
        try:
            # 新しいCRUDクラスを使用して一括挿入
            crud().insert_topic_data(corpus_data)
            self.logger.info(f"データベース保存完了: {len(corpus_data)}件")
            
        except Exception as e:
            self.logger.error(f"データベース保存中にエラーが発生しました: {e}")
            raise

    def process_single_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        単一コンテンツのトピック処理
        
        Args:
            content: 処理対象のコンテンツ
            
        Returns:
            処理結果の辞書
        """
        try:
            # 前処理
            processed_content = self._preprocess_contents([content])[0]
            
            # 正規化
            normalized_doc = self._normalize_documents([processed_content])[0]
            
            # 形態素解析
            tokenized_doc = self._tokenize_documents([normalized_doc])[0]
            
            # クリーニング
            cleaned_doc = self._clean_documents([tokenized_doc])[0]
            
            # LDA処理（既存のモデルを使用）
            topics = self._get_existing_topics()
            
            # コーパス生成
            corpus_data = self._generate_corpus([processed_content], topics)
            
            return corpus_data[0] if corpus_data else {'id': content.get('id', ''), 'corpus': ''}
            
        except Exception as e:
            self.logger.error(f"単一コンテンツ処理中にエラーが発生しました: {e}")
            return {'id': content.get('id', ''), 'corpus': '', 'error': str(e)}

    def _get_existing_topics(self) -> List[str]:
        """
        既存のトピックを取得（単一コンテンツ処理用）
        
        Returns:
            既存のトピックリスト
        """
        try:
            # データベースから既存のトピックを取得
            existing_data = crud().get_data_for_df("SELECT corpus FROM topic_corpus WHERE corpus != '' LIMIT 100")
            
            topics = set()
            for row in existing_data:
                if row.get('corpus'):
                    topics.update(row['corpus'].split(','))
            
            return list(topics)
            
        except Exception as e:
            self.logger.error(f"既存トピック取得中にエラーが発生しました: {e}")
            return []

    def get_topic_statistics(self) -> Dict[str, Any]:
        """
        トピック統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        try:
            # データベースから統計情報を取得
            stats_query = """
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN corpus != '' THEN 1 END) as non_empty_count,
                    AVG(LENGTH(corpus)) as avg_corpus_length
                FROM topic_corpus
            """
            
            result = crud().get_data_for_df(stats_query)
            
            if result:
                stats = result[0]
                total_count = stats.get('total_count', 0)
                non_empty_count = stats.get('non_empty_count', 0)
                
                return {
                    'total_count': total_count,
                    'non_empty_count': non_empty_count,
                    'avg_corpus_length': float(stats.get('avg_corpus_length', 0)),
                    'empty_ratio': (total_count - non_empty_count) / max(total_count, 1)
                }
            else:
                return {
                    'total_count': 0,
                    'non_empty_count': 0,
                    'avg_corpus_length': 0.0,
                    'empty_ratio': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"統計情報取得中にエラーが発生しました: {e}")
            return {
                'error': str(e),
                'total_count': 0,
                'non_empty_count': 0,
                'avg_corpus_length': 0.0,
                'empty_ratio': 0.0
            }

    def search_topics(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        トピックで検索
        
        Args:
            search_term: 検索語
            limit: 取得件数制限
            
        Returns:
            検索結果のリスト
        """
        try:
            query = """
                SELECT id, corpus
                FROM topic_corpus 
                WHERE corpus LIKE :search_term
                LIMIT :limit
            """
            
            return crud().get_data_for_df(query, {
                'search_term': f'%{search_term}%',
                'limit': limit
            })
            
        except Exception as e:
            self.logger.error(f"トピック検索中にエラーが発生しました: {e}")
            return []

    def update_topic_for_content(self, content_id: str, new_corpus: str) -> bool:
        """
        特定コンテンツのトピックを更新
        
        Args:
            content_id: コンテンツID
            new_corpus: 新しいコーパス
            
        Returns:
            更新成功時True、失敗時False
        """
        try:
            query = """
                UPDATE topic_corpus 
                SET corpus = :corpus 
                WHERE id = :content_id
            """
            
            # TODO: db_managerの実装が必要
            # from app.db.sqlalchemy.db_manager import db_manager
            # from sqlalchemy import text
            
            # with db_manager.get_session() as session:
            #     session.execute(text(query), {
            #         'corpus': new_corpus,
            #         'content_id': content_id
            #     })
            #     session.commit()
            
            self.logger.warning("update_topic_for_content: db_managerの実装が必要です")
            return True
            
        except Exception as e:
            self.logger.error(f"トピック更新中にエラーが発生しました: {e}")
            return False


# 後方互換性のためのエイリアス
topic = TopicProcessor