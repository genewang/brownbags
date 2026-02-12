# Project Structure Diagram

## Directory Tree

```
predictor/
│
├── api/                          # Flask Web Application
│   ├── api.py                    # Main Flask application
│   ├── requirements.txt          # Python dependencies
│   ├── templates/                # HTML templates
│   │   ├── base.html
│   │   └── index.html
│   └── static/                  # Static assets
│       └── css/
│           └── custom.css
│
├── model/                        # Machine Learning Components
│   ├── training.py              # Main training pipeline
│   ├── regression.py            # Regression model classes
│   ├── preprocessing.py         # Data preprocessing functions
│   ├── helpers.py               # Helper utilities
│   ├── first_glance.py          # Data exploration
│   ├── transfer.py             # Transfer learning
│   ├── importance.py           # Feature importance
│   ├── campaigns.csv            # Training data
│   ├── impressions.csv         # Impressions data
│   └── models/                  # Trained models storage
│       ├── impressions_model.pkl
│       └── impressions_columns.pkl
│
├── data/                        # Additional data files
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── system-diagrams.md
│   ├── architecture-overview.md
│   └── project-structure.md
│
├── Dockerfile.api               # API container definition
├── Dockerfile.model             # Model training container
├── docker-compose.yml           # Docker orchestration
├── Makefile                     # Build automation
├── .dockerignore                # Docker build exclusions
├── README.md                    # Project README
└── README-Docker.md             # Docker setup guide
```

## Component Relationships

```mermaid
graph TB
    subgraph "API Module"
        API_PY[api.py]
        Templates[templates/]
        Static[static/]
    end
    
    subgraph "Model Module"
        Training[training.py]
        Regression[regression.py]
        Preprocessing[preprocessing.py]
        Helpers[helpers.py]
        FirstGlance[first_glance.py]
    end
    
    subgraph "Data Files"
        CSV_Data[campaigns.csv]
        Models[models/*.pkl]
    end
    
    subgraph "Docker"
        Dockerfile_API[Dockerfile.api]
        Dockerfile_Model[Dockerfile.model]
        Compose[docker-compose.yml]
    end
    
    API_PY --> Templates
    API_PY --> Static
    API_PY --> Models
    
    Training --> Regression
    Training --> Preprocessing
    Training --> Helpers
    Training --> CSV_Data
    Training --> Models
    
    Regression --> Helpers
    Preprocessing --> Helpers
    FirstGlance --> CSV_Data
    
    Dockerfile_API --> API_PY
    Dockerfile_Model --> Training
    Compose --> Dockerfile_API
    Compose --> Dockerfile_Model
    
    style API_PY fill:#4CAF50
    style Training fill:#2196F3
    style Models fill:#FF9800
    style Compose fill:#9C27B0
```

## File Dependencies

### API Module
```
api.py
├── Flask framework
├── Templates (HTML)
├── Static assets (CSS)
└── Model files (.pkl)
    ├── impressions_model.pkl
    └── impressions_columns.pkl
```

### Model Module
```
training.py
├── regression.py
│   └── helpers.py
├── preprocessing.py
│   └── helpers.py
├── helpers.py
└── Data sources
    ├── campaigns.csv
    └── impressions.csv
```

## Data Flow Through Files

```mermaid
flowchart LR
    CSV[campaigns.csv] --> Training[training.py]
    Training --> Preprocess[preprocessing.py]
    Preprocess --> Regression[regression.py]
    Regression --> Helpers[helpers.py]
    Training --> Save[Save Models]
    Save --> PKL[models/*.pkl]
    
    PKL --> API[api.py]
    API --> Predict[Predictions]
    
    style CSV fill:#4CAF50
    style Training fill:#2196F3
    style PKL fill:#FF9800
    style API fill:#9C27B0
```

## Build and Deployment Flow

```mermaid
flowchart TD
    Source[Source Code] --> Build_API[Build API Image]
    Source --> Build_Model[Build Model Image]
    
    Build_API --> Dockerfile_API[Dockerfile.api]
    Build_Model --> Dockerfile_Model[Dockerfile.model]
    
    Dockerfile_API --> API_Image[API Container]
    Dockerfile_Model --> Model_Image[Model Container]
    
    API_Image --> Compose[docker-compose.yml]
    Model_Image --> Compose
    
    Compose --> Deploy[Deployed System]
    
    style Source fill:#4CAF50
    style Deploy fill:#2196F3
```

## Key Files Description

| File | Purpose |
|-----|---------|
| `api/api.py` | Main Flask application with routes and prediction logic |
| `model/training.py` | Complete ML training pipeline |
| `model/regression.py` | Regression model implementations (Linear, Tree, Forest, SVR) |
| `model/preprocessing.py` | Data cleaning, encoding, and scaling functions |
| `model/helpers.py` | Utility functions for accuracy calculation |
| `Dockerfile.api` | Container definition for API service |
| `Dockerfile.model` | Container definition for training environment |
| `docker-compose.yml` | Orchestration of containers |
| `Makefile` | Build and deployment automation |

## Module Responsibilities

### API Module
- **Purpose**: Serve predictions via web interface and REST API
- **Key Functions**: Request handling, data formatting, model loading, prediction
- **Dependencies**: Flask, pandas, scikit-learn, joblib

### Model Module
- **Purpose**: Train and evaluate ML models
- **Key Functions**: Data preprocessing, model training, evaluation, persistence
- **Dependencies**: pandas, scikit-learn, numpy, xgboost, statsmodels

### Docker Module
- **Purpose**: Containerize and orchestrate services
- **Key Files**: Dockerfiles, docker-compose.yml, Makefile
- **Benefits**: Isolation, reproducibility, easy deployment

