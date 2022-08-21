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