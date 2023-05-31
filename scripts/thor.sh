black ../src/
docker build -t thor:dev ../src/
docker rm -f thor-$1-$2-$3-$4|| true
docker run -v /disk2/mw4315/numerai-classic/$1/:/workspace/ --gpus device=$3 --user $(id -u):$(id -g) -it -d --rm --name thor-$1-$2-$3-$4  thor:dev
docker exec thor-$1-$2-$3-$4 python thor-$2.py > logs/output-$1-$2-$3-$4.log