"""
スケジューラーモジュール

このモジュールは、定期的なデータ処理タスクを実行するスケジューラーを提供します。
トピックコーパスの作成と関連記事スコアの計算を行います。
"""

import datetime
import logging
from typing import List, Dict, Any

from config import config
from app.lib.contents import microcms
from app.lib.topic import topic
from app.lib.related_score import create_score


# 設定値
MICROCMS_CONFIG = {
    'url': config.MICROCMS_URL,
    'api_key': config.MICROCMS_API_KEY,
    'limit': config.LIMIT
}


def init() -> None:
    """
    スケジューラーの初期化処理
    
    コンテンツデータの取得、トピックコーパスの作成、
    関連記事スコアの計算を順次実行します。
    
    Raises:
        Exception: 処理中にエラーが発生した場合
    """
    start_time = datetime.datetime.now()
    logging.info(f"スケジューラー開始: {start_time}")

    try:
        # コンテンツデータの取得
        content_list = _fetch_content_data()
        
        if not content_list:
            logging.warning("コンテンツデータが取得できませんでした")
            return

        # トピックコーパスの作成
        _create_topic_corpus(content_list)
        
        # 関連記事のスコア作成
        _create_related_scores(content_list)
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logging.info(f"スケジューラー完了: {end_time} (実行時間: {duration})")
        
    except Exception as e:
        logging.error(f"スケジューラー実行中にエラーが発生しました: {e}")
        raise


def _fetch_content_data() -> List[Dict[str, Any]]:
    """
    コンテンツデータを取得
    
    Returns:
        取得したコンテンツデータのリスト
        
    Raises:
        Exception: データ取得中にエラーが発生した場合
    """
    try:
        url = f"{MICROCMS_CONFIG['url']}?limit={MICROCMS_CONFIG['limit']}"
        api_key = MICROCMS_CONFIG['api_key']
        columns = ["id", "title", "blogContent"]
        
        logging.info("コンテンツデータを取得中...")
        raw_content_list = microcms.getData(url, api_key, columns)
        
        # リスト形式のデータを辞書形式に変換
        content_list = []
        for content in raw_content_list:
            if len(content) >= 3:  # id, title, blogContentの3つの要素があることを確認
                content_dict = {
                    'id': content[0],
                    'title': content[1],
                    'blogContent': content[2]
                }
                content_list.append(content_dict)
        
        logging.info(f"コンテンツデータ取得完了: {len(content_list)}件")
        return content_list
        
    except Exception as e:
        logging.error(f"コンテンツデータ取得中にエラーが発生しました: {e}")
        raise


def _create_topic_corpus(content_list: List[Dict[str, Any]]) -> None:
    """
    トピックコーパスを作成
    
    Args:
        content_list: 処理対象のコンテンツリスト
        
    Raises:
        Exception: トピックコーパス作成中にエラーが発生した場合
    """
    try:
        logging.info("トピックコーパスを作成中...")
        topic().start(content_list)
        logging.info("トピックコーパス作成完了")
        
    except Exception as e:
        logging.error(f"トピックコーパス作成中にエラーが発生しました: {e}")
        raise


def _create_related_scores(content_list: List[Dict[str, Any]]) -> None:
    """
    関連記事スコアを作成
    
    Args:
        content_list: 処理対象のコンテンツリスト
        
    Raises:
        Exception: 関連記事スコア作成中にエラーが発生した場合
    """
    try:
        logging.info("関連記事スコアを作成中...")
        create_score().start(content_list)
        logging.info("関連記事スコア作成完了")
        
    except Exception as e:
        logging.error(f"関連記事スコア作成中にエラーが発生しました: {e}")
        raise
