build:
	docker compose build

run:
	docker compose up

down:
	docker compose down

rebuild:
	docker compose down
	docker compose build --no-cache