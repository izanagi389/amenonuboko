"""
microCMSコンテンツ取得モジュール

このモジュールは、microCMS APIからコンテンツデータを取得し、
テキストの前処理を行う機能を提供します。
"""

import requests
from typing import List, Any, Union

from app.lib.text.shape import TextShapping


def getData(url: str, api_key: str, columns: List[str]) -> List[Union[List[Any], str]]:
    """
    microCMS APIからコンテンツデータを取得
    
    Args:
        url: APIエンドポイントURL
        api_key: APIキー
        columns: 取得するカラム名のリスト
        
    Returns:
        取得したコンテンツデータのリスト。エラーの場合はエラーメッセージを含むリスト
        
    Raises:
        requests.exceptions.RequestException: APIリクエスト中にエラーが発生した場合
    """
    headers = {'X-API-KEY': api_key}
    contents = []
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        for content in data["contents"]:
            content_list = []
            for column in columns:
                if isinstance(content[column], list):
                    # リスト形式のコンテンツを結合
                    combined_text = ""
                    for item in content[column]:
                        if item.get('content') is not None:
                            combined_text += item["content"]

                    # テキストの前処理
                    processed_text = TextShapping.remove_html_tag(combined_text)
                    processed_text = TextShapping.remove_url(processed_text)
                    content_list.append(processed_text)
                else:
                    content_list.append(content[column])

            contents.append(content_list)

    except requests.exceptions.RequestException as e:
        print(f"エラー: {e}")
        return [str(e)]

    return contents
