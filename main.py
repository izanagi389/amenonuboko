from app import app
from api.related_handler import related_doc_list
from flask_cors import CORS
from app.scheduler import scheduler
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from app.lib.db.title_model import init_data, cerate_title_table


app.register_blueprint(related_doc_list)
CORS(app)

scheduler()
sched = BackgroundScheduler(daemon=True)
sched.add_job(scheduler, 'interval', days=1)

sched.start()
atexit.register(lambda: sched.shutdown())

app.config.from_object('app.lib.db.config.Config')

cerate_title_table()
init_data()


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
