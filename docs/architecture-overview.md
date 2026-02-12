# System Architecture Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Web Browser          │          API Clients                    │
│  (User Interface)      │          (REST API)                     │
└────────────┬──────────┴──────────────┬──────────────────────────┘
             │                         │
             │ HTTP Requests           │ JSON Requests
             │                         │
┌────────────▼─────────────────────────▼──────────────────────────┐
│                    APPLICATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         Flask Web Application (Port 5001)                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │   Routes     │  │   Handlers    │  │  Templates   │   │  │
│  │  │  /, /ping    │  │  /:metric     │  │  index.html  │   │  │
│  │  │  /campaign   │  │  /campaign    │  │  static/css  │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Function Calls
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ Prediction Engine │  │  Data Formatter  │                   │
│  │  - predict()      │  │  - format_cat()  │                   │
│  │  - predict_met()  │  │  - align_feat()  │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                   │
│  ┌──────────────────┐                                           │
│  │   Model Loader    │                                           │
│  │  - load_model()   │                                           │
│  │  - load_columns()  │                                           │
│  └──────────────────┘                                           │
│                                                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ File I/O
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      MODEL LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐  ┌──────────────────────┐          │
│  │  impressions_model    │  │  impressions_columns │          │
│  │      .pkl             │  │      .pkl             │          │
│  │  (Trained ML Model)   │  │  (Feature Schema)     │          │
│  └──────────────────────┘  └──────────────────────┘          │
│                                                                   │
│  Model Types: Linear, Decision Tree, Random Forest, SVR          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING ENVIRONMENT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Data Load    │→ │ Preprocessing│→ │   Training   │         │
│  │  campaigns.csv│  │  - Clean     │  │  - Build      │         │
│  └──────────────┘  │  - Encode     │  │  - Evaluate   │         │
│                    │  - Scale      │  │  - Select     │         │
│                    └──────────────┘  │  - Save        │         │
│                                      └──────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Relationships

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       │ 1. Submit Campaign Data
       ▼
┌─────────────────────────────────────┐
│         Flask Application            │
│  ┌───────────────────────────────┐  │
│  │  Request Handler               │  │
│  │  - Parse input                 │  │
│  │  - Format categoricals         │  │
│  └──────────────┬────────────────┘  │
│                 │                     │
│  ┌──────────────▼────────────────┐  │
│  │  Prediction Engine             │  │
│  │  - Load model                  │  │
│  │  - Align features              │  │
│  │  - Run prediction              │  │
│  │  - Calculate metrics           │  │
│  └──────────────┬────────────────┘  │
└─────────────────┼────────────────────┘
                  │
                  │ 2. Load Model Files
                  ▼
┌─────────────────────────────────────┐
│         Model Storage                │
│  - impressions_model.pkl             │
│  - impressions_columns.pkl           │
└─────────────────────────────────────┘
                  │
                  │ 3. Return Predictions
                  ▼
┌─────────────────────────────────────┐
│         Response                    │
│  - Impressions (low/high)            │
│  - Clicks (low/high)                 │
│  - Purchases (low/high)              │
└─────────────────────────────────────┘
```

## Data Flow Summary

1. **Input**: User submits campaign parameters via web form or API
2. **Processing**: Flask app formats data and loads ML model
3. **Prediction**: Model generates predictions for impressions/clicks/purchases
4. **Output**: Results displayed with confidence ranges (±20%)

## Technology Stack

- **Web Framework**: Flask 2.3.3
- **ML Library**: Scikit-learn 1.3.0
- **Data Processing**: Pandas 2.0.3, NumPy 1.24.3
- **Model Persistence**: Joblib 1.3.2
- **Containerization**: Docker, Docker Compose
- **Language**: Python 3.11

