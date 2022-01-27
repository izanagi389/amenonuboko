from apscheduler.schedulers.blocking import BlockingScheduler

import app.touch_model as transformers_model_lib
from app import crud

def init():
    transformers_model_lib.initialize_cl_tohoku_load()
    crud.init_data()


scheduler = BlockingScheduler()

scheduler.add_job(init, 'interval', days=1)

