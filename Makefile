.PHONY: frontend server all clean test docker

frontend:
	cd frontend/sehat-ui && npm run dev

server:
	cd app && uvicorn main:combined_app --reload --port 8000

all:
	make -j2 frontend server

clean:
	echo "clean not implemented yet"

test:
	echo "run tests not implemented yet"

docker:
	docker volume create redis-stack-data
	docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 -v redis-stack-data:/data redis/redis-stack:latest
