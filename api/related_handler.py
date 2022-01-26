from flask import Blueprint
import json

from app.lib.db.title_model import getJsonData

related_doc_list = Blueprint(
    'related_doc_list', __name__, url_prefix='/related_doc_list')


@related_doc_list.route('/<serach_id>', methods=['GET'])
def related(serach_id):

    serach_id = serach_id

    if (serach_id == None):
        return json.dumps(["パラメータがありません！"], indent=2, ensure_ascii=False)

    result_data = getJsonData(serach_id, 5)

    return result_data
