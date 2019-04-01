# Run this after copying proper configuration
cd redis-stable
redis-server redis.conf --loadmodule ../RedisAI/build/redisai.so &
redis-server sentinel.conf --sentinel &
cd ~