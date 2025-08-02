"""
Sudachi形態素解析モジュール

このモジュールは、Sudachiを使用した日本語の形態素解析機能を提供します。
"""

from typing import List, Tuple

from sudachipy import dictionary, tokenizer


def Tokenize(docs: List[str], hinshi_list: List[str] = ["名詞", "", "一般"]) -> List[List[str]]:
    """
    ドキュメントを形態素解析してトークン化
    
    Args:
        docs: 解析対象のドキュメントリスト
        hinshi_list: 品詞フィルタリングリスト
        
    Returns:
        トークン化されたドキュメントリスト
    """
    tokenizer_obj = dictionary.Dictionary(dict_type="full").create()

    # ドキュメントをトークンに分割
    mode = tokenizer.Tokenizer.SplitMode.C
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # 小文字に変換
        docs[idx] = [m.surface() for m in tokenizer_obj.tokenize(docs[idx], mode)
                     if check_for_hinshi(m.part_of_speech(), hinshi_list)]  # 単語に分割

    return docs


def check_for_hinshi(part_of_speech: Tuple[str, ...], hinshi: List[str]) -> bool:
    """
    品詞チェック
    
    Args:
        part_of_speech: 品詞情報のタプル
        hinshi: チェック対象の品詞リスト
        
    Returns:
        品詞が一致する場合True、そうでなければFalse
    """
    part_of_speech_list = list(part_of_speech)
    flag = 0

    for i, h in enumerate(hinshi):
        if h == "":
            flag += 1
            continue
        if not part_of_speech_list[i] == h:
            return False

    if len(hinshi) == flag:
        return False
    else:
        return True
