"""
SentenceBERT日本語モデルモジュール

このモジュールは、日本語テキストの文埋め込みを生成するための
SentenceBERTモデルを提供します。
"""

from typing import List, Optional, Union

import torch
from transformers import BertJapaneseTokenizer, BertModel


class SentenceBertJapanese:
    """
    SentenceBERT日本語モデル
    
    日本語テキストをベクトル化するためのSentenceBERTモデルです。
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """
        SentenceBertJapaneseクラスの初期化
        
        Args:
            model_name_or_path: モデル名またはパス
            device: 使用するデバイス（cuda/cpu）
        """
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        平均プーリングを実行
        
        Args:
            model_output: モデルの出力
            attention_mask: アテンションマスク
            
        Returns:
            平均プーリングされた埋め込み
        """
        # モデル出力の最初の要素にはすべてのトークン埋め込みが含まれる
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        文をベクトル化
        
        Args:
            sentences: ベクトル化する文のリスト
            batch_size: バッチサイズ
            
        Returns:
            文の埋め込みベクトル
        """
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, 
                padding="longest",
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, 
                encoded_input["attention_mask"]
            ).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
