"""
microCMS API モジュール

このモジュールは、microCMS APIからデータを取得する機能を提供します。
"""

import requests
from typing import List, Any, Union


def getData(url: str, api_key: str, columns: List[str]) -> List[Union[List[Any], str]]:
    """
    microCMS APIからデータを取得
    
    Args:
        url: APIエンドポイントURL
        api_key: APIキー
        columns: 取得するカラム名のリスト
        
    Returns:
        取得したデータのリスト。エラーの場合はエラーメッセージを含むリスト
        
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
                content_list.append(content[column])

            contents.append(content_list)

    except requests.exceptions.RequestException as e:
        print(f"エラー: {e}")
        return [str(e)]

    return contents
