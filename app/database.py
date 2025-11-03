from sqlmodel import create_engine, SQLModel

DATABASE_URL = "postgresql://sehat_user:sehat%40123@localhost:5432/sehat_db"

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)


#To login into terminal: psql -U sehat_user -d sehat_db -h localhost -W 