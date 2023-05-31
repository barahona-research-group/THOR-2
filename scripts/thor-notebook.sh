black ../src/
docker build -t thor:dev ../src/
docker rm -f thor-notebook || true
docker run -v /disk2/mw4315/:/workspace/ --gpus device=$1  -it -p 8890:8890 -d --rm --name thor-notebook thor:dev
docker exec -d thor-notebook jupyter notebook --ip 0.0.0.0 --port 8890 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' &