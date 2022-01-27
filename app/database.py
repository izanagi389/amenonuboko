from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config

SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4'.format(**{
    'user': config.MYSQL_USER,
    'password': config.MYSQL_PASSWORD,
    'host': config.MYSQL_HOST,
    'database': config.MYSQL_DATABASE
})

engine = create_engine(
    SQLALCHEMY_DATABASE_URI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
