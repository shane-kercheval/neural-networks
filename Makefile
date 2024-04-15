####
# DOCKER
####
docker_build:
	docker compose -f docker-compose.yml build

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	docker compose -f docker-compose.yml build --no-cache

####
# Project
####
linting:
	ruff check torchpy
	ruff check torchpy_tests

unit_tests:
	pytest torchpy_tests

tests: linting unit_tests
