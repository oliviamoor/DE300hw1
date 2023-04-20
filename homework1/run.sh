docker build -t hw1 .

docker run -v $(pwd)/images:/app/images -it hw1 