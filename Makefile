.PHONY: frontend server all clean test

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

