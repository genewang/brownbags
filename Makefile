.PHONY: help build run stop clean logs train shell

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build all Docker images"
	@echo "  run       - Start the API service"
	@echo "  stop      - Stop all services"
	@echo "  clean     - Remove containers and images"
	@echo "  logs      - Show API service logs"
	@echo "  train     - Run model training"
	@echo "  shell     - Open shell in API container"
	@echo "  shell-model - Open shell in model container"

# Build all images
build:
	docker-compose build

# Start the API service
run:
	docker-compose up -d api

# Stop all services
stop:
	docker-compose down

# Clean up containers and images
clean:
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Show logs
logs:
	docker-compose logs -f api

# Run model training
train:
	docker-compose --profile training run --rm model-training python training.py

# Open shell in API container
shell:
	docker-compose exec api /bin/bash

# Open shell in model container
shell-model:
	docker-compose --profile training run --rm -it model-training /bin/bash

# Run first glance analysis
first-glance:
	docker-compose --profile training run --rm model-training python first_glance.py

# Run regression analysis
regression:
	docker-compose --profile training run --rm model-training python regression.py
