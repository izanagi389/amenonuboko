"""
LDA（Latent Dirichlet Allocation）処理モジュール

このモジュールは、トピックモデリングのためのLDA処理を提供します。
"""

from typing import List, Any, Optional
from gensim.models import LdaModel, Phrases
from gensim.corpora import Dictionary


class LDA:
    """
    LDA（Latent Dirichlet Allocation）処理クラス
    
    トピックモデリングのためのLDA処理を実行するクラスです。
    """

    @staticmethod
    def load(dictionary: Dictionary, corpus: List[List[tuple]], 
             num_topics: int = 5, passes: int = 10, iterations: int = 400, 
             eval_every: Optional[int] = None) -> LdaModel:
        """
        LDAモデルを読み込みまたは作成
        
        Args:
            dictionary: 辞書オブジェクト
            corpus: コーパスデータ
            num_topics: トピック数
            passes: パス数
            iterations: 反復回数
            eval_every: 評価間隔
            
        Returns:
            LDAモデル
        """
        # 辞書をロード（インデックス作成のため）
        temp = dictionary[0]  # 辞書をロード
        id2word = dictionary.id2token

        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )

        return model

    @staticmethod
    def bigram2docs(docs: List[List[str]], min_count: int = 20) -> List[List[str]]:
        """
        バイグラムをドキュメントに変換
        
        Args:
            docs: ドキュメントリスト
            min_count: 最小出現回数
            
        Returns:
            バイグラム追加済みドキュメントリスト
        """
        bigram = Phrases(docs, min_count)
        
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # トークンがバイグラムの場合、ドキュメントに追加
                    docs[idx].append(token)

        return docs

    @staticmethod
    def get_dictionary(docs: List[List[str]]) -> Dictionary:
        """
        辞書を作成
        
        Args:
            docs: ドキュメントリスト
            
        Returns:
            作成された辞書
        """
        # 辞書を作成
        dictionary = Dictionary(docs)

        # 稀すぎる単語（5文書未満）と頻出すぎる単語（50%以上の文書）を除外
        dictionary.filter_extremes(no_below=5, no_above=0.5)

        return dictionary

    @staticmethod
    def get_corpus(dictionary: Dictionary, docs: List[List[str]]) -> List[List[tuple]]:
        """
        コーパスを作成
        
        Args:
            dictionary: 辞書オブジェクト
            docs: ドキュメントリスト
            
        Returns:
            作成されたコーパス
        """
        # ドキュメントのBag-of-words表現
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        return corpus

    @staticmethod
    def topics(model: LdaModel, corpus: List[List[tuple]]) -> List[str]:
        """
        トピックを抽出
        
        Args:
            model: LDAモデル
            corpus: コーパスデータ
            
        Returns:
            抽出されたトピックリスト
        """
        top_topics = model.top_topics(corpus)

        topic_list = []

        for topic in top_topics:
            for tt in topic:
                if isinstance(tt, list):
                    for t in tt:
                        topic_list.append(t[1])

        # 重複を除去
        topic_list = list(set(topic_list))

        return topic_list
