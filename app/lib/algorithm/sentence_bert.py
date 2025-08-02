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
        try:
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
            self.model = BertModel.from_pretrained(model_name_or_path)
            self.model.eval()

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            
            # メタテンソルエラーを回避するため、to_empty()を使用
            try:
                if hasattr(self.model, 'to_empty'):
                    self.model = self.model.to_empty(device=self.device)
                else:
                    self.model = self.model.to(self.device)
            except Exception as e:
                # GPU移動に失敗した場合はCPUを使用
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                
        except Exception as e:
            raise Exception(f"SentenceBERTモデルの初期化に失敗しました: {e}")

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
        if not sentences:
            return torch.tensor([])
        
        # 入力の検証
        valid_sentences = []
        for sentence in sentences:
            if sentence and isinstance(sentence, str) and sentence.strip():
                valid_sentences.append(sentence.strip())
        
        if not valid_sentences:
            return torch.tensor([])
        
        try:
            all_embeddings = []
            iterator = range(0, len(valid_sentences), batch_size)
            
            for batch_idx in iterator:
                batch = valid_sentences[batch_idx:batch_idx + batch_size]
                
                if not batch:
                    continue

                try:
                    encoded_input = self.tokenizer.batch_encode_plus(
                        batch, 
                        padding="longest",
                        truncation=True, 
                        return_tensors="pt",
                        max_length=512  # 最大長を制限
                    ).to(self.device)
                    
                    model_output = self.model(**encoded_input)
                    sentence_embeddings = self._mean_pooling(
                        model_output, 
                        encoded_input["attention_mask"]
                    ).to('cpu')

                    all_embeddings.append(sentence_embeddings)
                    
                except Exception as e:
                    import logging
                    logging.error(f"バッチ処理エラー: {e}")
                    continue

            if all_embeddings:
                return torch.cat(all_embeddings, dim=0)
            else:
                return torch.tensor([])
                
        except Exception as e:
            import logging
            logging.error(f"encode処理エラー: {e}")
            return torch.tensor([])
