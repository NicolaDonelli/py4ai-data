from mongomock_motor import AsyncMongoMockClient, AsyncMongoMockCollection


async def init_mongo(
    db_name: str, collection: str, cleanup: bool = True
) -> AsyncMongoMockCollection:
    client = AsyncMongoMockClient(
        "mongodb://user:pass@host:27017", connectTimeoutMS=250
    )
    mongo_collection: AsyncMongoMockCollection = client[db_name][collection]
    if cleanup is True:
        await mongo_collection.delete_many({})
    return mongo_collection
