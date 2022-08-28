from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config
import pandas as pd
from pandas.core.frame import DataFrame


class crud():

    def __init__(self):
        self.SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4'.format(**{
            'user': config.MYSQL_USER,
            'password': config.MYSQL_PASSWORD,
            'host': config.MYSQL_HOST,
            'database': config.MYSQL_DATABASE
        })

        self.engine = create_engine(
            self.SQLALCHEMY_DATABASE_URI
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine)

        self.Base = declarative_base()

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def get_data_for_df(self, query_string: str) -> DataFrame:
        df = pd.read_sql(query_string, con=self.engine)

        return df

    def add_data_for_df(self, df: DataFrame, table_name: str):
        df.to_sql(table_name, self.engine, index=False, if_exists='replace')

    def get_topic_data_for_df(self, search_id: str):
        database = config.MYSQL_DATABASE
        table_name = "topic_corpus"
        query = "select id, corpus from {}.{} where id = '{}'".format(
            database, table_name, search_id)
        df = pd.read_sql(query, con=self.engine)

        return df.to_dict(orient='records')[0]

    def get_relate_score_for_df(self, search_id: str):
        database = config.MYSQL_DATABASE
        table_name = "related_data_v2"
        query = "select id, relate_id, relate_title, bert_cos_distance from {}.{} where id = '{}'".format(
            database, table_name, search_id)
        df = pd.read_sql(query, con=self.engine)

        return df
