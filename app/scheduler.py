import app.touch_model as transformers_model_lib
from app import crud

def init():
    print("get model data")
    transformers_model_lib.initialize_cl_tohoku_load()
    print("got model")
    print("init database")
    crud.init_data()
    print("finish")

