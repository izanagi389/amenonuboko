from flask import Flask
from api.related_handler import related_doc_list
from flask_cors import CORS
from lib.model_creating import *
from apscheduler.schedulers.background import BackgroundScheduler
import atexit


app = Flask(__name__)

app.register_blueprint(related_doc_list)
CORS(app)

def scheduler():
    print("Create Model")
    doc2vec_model_create()


sched = BackgroundScheduler(daemon=True)
sched.add_job(scheduler, 'interval', days=1)
sched.start()

atexit.register(lambda: sched.shutdown())

if __name__ == '__main__':
    scheduler()
    app.run(host='0.0.0.0', threaded=True)
