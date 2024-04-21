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
# torchpy - Pytorch-like library
####
torchpy_linting:
	ruff check torchpy
	ruff check torchpy_tests

torchpy_unit_tests:
	pytest torchpy_tests

torchpy_tests: linting unit_tests

####
# torchcpp - Pytorch-like library for C++
####
