import redis


class RedisConnector:
    def __init__(self, host="redis", port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_database = redis.Redis(host=self.host, port=self.port, db=self.db)

    def set(self, key, value, ex=None):
        return self.redis_database.set(key, value, ex=ex)

    def get(self, key):
        value = self.redis_database.get(key)
        return value.decode("utf-8") if value else None

    def delete(self, *keys):
        return self.redis_database.delete(*keys)

    def exists(self, key):
        return self.redis_database.exists(key)

    def expire(self, key, seconds):
        return self.redis_database.expire(key, seconds)

    def keys(self, pattern="*"):
        return [key.decode("utf-8") for key in self.redis_database.keys(pattern)]

    def flushdb(self):
        return self.redis_database.flushdb()

    def hset(self, key, mapping=None, **kwargs):
        if mapping:
            return self.redis_database.hset(key, mapping=mapping)
        return self.redis_database.hset(key, mapping=kwargs)

    def hget(self, key, field):
        value = self.redis_database.hget(key, field)
        return value.decode("utf-8") if value else None

    def hgetall(self, key):
        record = self.redis_database.hgetall(key)
        return {k.decode("utf-8"): v.decode("utf-8") for k, v in record.items()}

    def hdel(self, key, *fields):
        return self.redis_database.hdel(key, *fields)

    def hexists(self, key, field):
        return self.redis_database.hexists(key, field)
