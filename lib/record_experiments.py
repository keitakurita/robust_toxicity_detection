from typing import *
import pymongo
from bson.objectid import ObjectId
import os
import logging

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

PASSWORD = "mongo11747" #  not good practice, but this is a private repo so...

conn_str = f"mongodb+srv://root:{PASSWORD}@cluster0-ptgoc.mongodb.net/test?retryWrites=true"

client = pymongo.MongoClient(conn_str)
db = client.experiments
collection = db.logs

def _cln(v: Any) -> Any:
    """Ensure variables are serializable"""
    if isinstance(v, (np.float, np.float16, np.float32, np.float64, np.float128)):
        return float(v)
    elif isinstance(v, (np.int, np.int0, np.int8, np.int16, np.int32, np.int64)):
        return int(v)
    elif isinstance(v, dict):
        return {k: _cln(v_) for k, v_ in v.items()}
    else:
        return v

def record(log: dict):
    res = collection.insert_one({str(k): _cln(v) for k, v in log.items()})
    logger.info(f"Inserted results at id {res.inserted_id}")
    return res

def find(id_: Optional[str]=None, query: Optional[dict]=None):
    if query is None: query = {"_id": ObjectId(id_)}
    res = collection.find_one(query)
    return res

def delete(id_: Optional[str]=None, query: Optional[dict]=None):
    if query is None: query = {"_id": ObjectId(id_)}
    res = collection.delete_many(query)
    logger.info(f"Deleted {res.deleted_count} entries")
    return res

if __name__ == "__main__":
    res = record({"hoge": "foo"})
    delete(id_=res.inserted_id)
