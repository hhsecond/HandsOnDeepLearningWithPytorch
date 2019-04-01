sudo apt update 
sudo apt install -y build-essential tcl libjemalloc-dev git cmake unzip

sudo ufw allow 6379
sudo ufw allow 26379

curl -O http://download.redis.io/redis-stable.tar.gz
tar xzvf redis-stable.tar.gz
cd redis-stable
make
sudo make install
cd ~
rm redis-stable.tar.gz

git clone https://github.com/RedisAI/RedisAI.git
cd RedisAI
bash get_deps.sh cpu
mkdir build
cd build
cmake -DDEPS_PATH=../deps/install ..
make
cd ~
