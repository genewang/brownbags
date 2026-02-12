# Docker Setup for Predictor Project

This project includes Docker configuration for easy deployment and development.

## Prerequisites

- Docker
- Docker Compose

## Quick Start

### 1. Build the images
```bash
make build
# or
docker-compose build
```

### 2. Start the API service
```bash
make run
# or
docker-compose up -d api
```

The API will be available at `http://localhost:5000`

### 3. Check the service
```bash
curl http://localhost:5000/ping
```

## Available Commands

Use the Makefile for common operations:

```bash
make help          # Show all available commands
make build         # Build all Docker images
make run           # Start the API service
make stop          # Stop all services
make clean         # Remove containers and images
make logs          # Show API service logs
make train         # Run model training
make shell         # Open shell in API container
make shell-model   # Open shell in model container
make first-glance  # Run first glance analysis
make regression    # Run regression analysis
```

## Services

### API Service
- **Port**: 5000
- **Purpose**: Flask web application for making predictions
- **Health Check**: Available at `/ping` endpoint
- **Auto-restart**: Enabled

### Model Training Service
- **Purpose**: Environment for training and analyzing ML models
- **Profile**: Only started when explicitly needed
- **Usage**: Run specific scripts like `training.py`, `first_glance.py`, etc.

## Development

### Running Model Training
```bash
# Train models
make train

# Run data analysis
make first-glance

# Run regression analysis
make regression
```

### Interactive Shell
```bash
# API container
make shell

# Model container
make shell-model
```

### Viewing Logs
```bash
make logs
```

## Data Persistence

- Model files are mounted from `./model/models/` to the API container
- Training data is mounted from `./model/` to the model container
- Changes to model files are reflected immediately in the API

## Stopping Services

```bash
make stop
# or
docker-compose down
```

## Cleanup

To completely remove all containers, images, and volumes:

```bash
make clean
```

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "5001:5000"  # Use port 5001 on host
```

### Permission Issues
The containers run as non-root users for security. If you encounter permission issues, check file ownership in the mounted volumes.

### Memory Issues
For large model training, you may need to increase Docker memory limits in Docker Desktop settings.

## Customization

### Environment Variables
Modify environment variables in `docker-compose.yml` or create a `.env` file.

### Dependencies
Update `requirements.txt` files and rebuild images:

```bash
make build
```

### Model Updates
The API service automatically picks up new model files from the mounted `./model/models/` directory.
