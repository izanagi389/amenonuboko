import app.lib.transformers.model as transformers_model_lib
from app.lib.db.title_model import init_data

def scheduler():
    transformers_model_lib.initialize_cl_tohoku_load()
    init_data()





