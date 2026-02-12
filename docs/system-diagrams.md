# Marketing Performance Predictor - System Diagrams

This document contains comprehensive system diagrams for the Marketing Performance Predictor project.

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        API_Client[API Client]
    end
    
    subgraph "Application Layer"
        Flask_App[Flask Web Application<br/>Port: 5001]
        Templates[HTML Templates]
        Static[Static Assets<br/>CSS/JS]
    end
    
    subgraph "Business Logic Layer"
        Prediction_Engine[Prediction Engine]
        Data_Formatter[Data Formatter]
        Model_Loader[Model Loader]
    end
    
    subgraph "Model Layer"
        ML_Models[ML Models<br/>.pkl files]
        Model_Columns[Feature Columns<br/>.pkl files]
    end
    
    subgraph "Training Environment"
        Training_Pipeline[Training Pipeline]
        Preprocessing[Data Preprocessing]
        Regression[Regression Models]
        Evaluation[Model Evaluation]
    end
    
    subgraph "Data Layer"
        CSV_Data[Campaign Data<br/>CSV Files]
        Models_Dir[Models Directory]
    end
    
    Browser --> Flask_App
    API_Client --> Flask_App
    Flask_App --> Templates
    Flask_App --> Static
    Flask_App --> Prediction_Engine
    Prediction_Engine --> Data_Formatter
    Prediction_Engine --> Model_Loader
    Model_Loader --> ML_Models
    Model_Loader --> Model_Columns
    ML_Models --> Models_Dir
    Model_Columns --> Models_Dir
    
    Training_Pipeline --> Preprocessing
    Preprocessing --> CSV_Data
    Preprocessing --> Regression
    Regression --> Evaluation
    Evaluation --> ML_Models
    
    style Flask_App fill:#4CAF50
    style Prediction_Engine fill:#2196F3
    style ML_Models fill:#FF9800
    style Training_Pipeline fill:#9C27B0
```

## 2. Data Flow Diagram

```mermaid
flowchart TD
    Start([User Input]) --> Web_Form[Web Form<br/>Budget, Dates, Channels, etc.]
    
    Web_Form --> Format_Data[Format Input Data]
    Format_Data --> Categorical[Convert Categoricals]
    Categorical --> Feature_Vector[Create Feature Vector]
    
    Feature_Vector --> Load_Model[Load ML Model]
    Load_Model --> Model_File[impressions_model.pkl]
    Load_Model --> Columns_File[impressions_columns.pkl]
    
    Model_File --> Align_Features[Align Features with Model]
    Columns_File --> Align_Features
    
    Align_Features --> Predict[Model Prediction]
    Predict --> Direct_Pred[Direct Prediction]
    
    Direct_Pred --> Calculate_CPX[Calculate CPX]
    Calculate_CPX --> Transfer_Model[Transfer Model Prediction]
    Transfer_Model --> Final_Pred[Final Predictions]
    
    Final_Pred --> Round_Values[Round & Format Values]
    Round_Values --> Confidence_Range[Calculate Confidence Ranges<br/>±20%]
    Confidence_Range --> Display[Display Results]
    
    Display --> End([User Sees Results])
    
    style Start fill:#4CAF50
    style Predict fill:#2196F3
    style Final_Pred fill:#FF9800
    style End fill:#4CAF50
```

## 3. API Endpoints Structure

```mermaid
graph LR
    subgraph "Flask Application"
        Root[GET/POST /<br/>Main Web Interface]
        Ping[GET/POST /ping<br/>Health Check]
        Metric[POST /:metric<br/>Single Metric Prediction]
        Campaign[POST /campaign<br/>Full Campaign Prediction]
    end
    
    subgraph "Handler Functions"
        Root_Handler[root Handler]
        Ping_Handler[ping Handler]
        Metric_Handler[metric_prediction]
        Campaign_Handler[campaign_prediction]
    end
    
    subgraph "Core Functions"
        Format[format_categoricals]
        Predict_Metrics[predict_metrics]
        Predict[predict]
        Load_Model[load_from_bucket]
    end
    
    Root --> Root_Handler
    Ping --> Ping_Handler
    Metric --> Metric_Handler
    Campaign --> Campaign_Handler
    
    Root_Handler --> Format
    Root_Handler --> Predict_Metrics
    Metric_Handler --> Format
    Metric_Handler --> Predict
    Campaign_Handler --> Format
    Campaign_Handler --> Predict_Metrics
    
    Predict_Metrics --> Predict
    Predict --> Load_Model
    
    style Root fill:#4CAF50
    style Ping fill:#81C784
    style Metric fill:#2196F3
    style Campaign fill:#2196F3
    style Load_Model fill:#FF9800
```

## 4. Model Training Pipeline

```mermaid
flowchart TD
    Start([Training Start]) --> Load_Data[Load Campaign Data<br/>campaigns.csv]
    
    Load_Data --> Trim_Data[Trim Data<br/>Remove Unused Metrics]
    Trim_Data --> Preprocess[Data Preprocessing Pipeline]
    
    Preprocess --> Data_Pipeline[data_pipeline Function]
    Data_Pipeline --> Filter[Filter Valid Records]
    Filter --> Drop_Cols[Drop High-Missing Columns]
    Drop_Cols --> Create_Buckets[Create 'Other' Buckets]
    Create_Buckets --> One_Hot[One-Hot Encoding]
    
    One_Hot --> Split[Split Pipeline]
    Split --> Train_Test[Train/Test Split<br/>80/20]
    Train_Test --> Scale[Feature Scaling]
    
    Scale --> Build_Models[Build Regression Models]
    Build_Models --> Linear[Linear Regression]
    Build_Models --> Tree[Decision Tree]
    Build_Models --> Forest[Random Forest]
    Build_Models --> SVR[Support Vector Regressor]
    
    Linear --> Evaluate[Evaluate Models]
    Tree --> Evaluate
    Forest --> Evaluate
    SVR --> Evaluate
    
    Evaluate --> Train_Acc[Calculate Training Accuracy]
    Evaluate --> Test_Acc[Calculate Test Accuracy]
    
    Train_Acc --> Select_Best[Select Best Model]
    Test_Acc --> Select_Best
    
    Select_Best --> Fit_Full[Fit on Full Dataset]
    Fit_Full --> Save_Model[Save Model & Columns]
    Save_Model --> Model_File[impressions_model.pkl]
    Save_Model --> Columns_File[impressions_columns.pkl]
    
    Model_File --> End([Training Complete])
    Columns_File --> End
    
    style Start fill:#4CAF50
    style Preprocess fill:#2196F3
    style Evaluate fill:#FF9800
    style Save_Model fill:#9C27B0
    style End fill:#4CAF50
```

## 5. Docker Deployment Architecture

```mermaid
graph TB
    subgraph "Host Machine"
        Docker_Compose[Docker Compose]
    end
    
    subgraph "Docker Network: predictor_default"
        subgraph "API Container"
            API_Image[python:3.11-slim]
            Flask_App[Flask Application]
            API_Volumes[Volumes:<br/>./model/models → /app/models<br/>./model → /app/data]
        end
        
        subgraph "Model Training Container"
            Model_Image[python:3.11-slim]
            Training_Env[Training Environment]
            Model_Volumes[Volumes:<br/>./model → /app<br/>./data → /app/data]
        end
    end
    
    subgraph "Host File System"
        Model_Dir[./model/models/]
        Model_Code[./model/]
        API_Code[./api/]
        Data_Dir[./data/]
    end
    
    Docker_Compose --> API_Image
    Docker_Compose --> Model_Image
    
    API_Image --> Flask_App
    Model_Image --> Training_Env
    
    Flask_App --> API_Volumes
    Training_Env --> Model_Volumes
    
    API_Volumes --> Model_Dir
    API_Volumes --> Model_Code
    Model_Volumes --> Model_Code
    Model_Volumes --> Data_Dir
    
    Flask_App -.->|Port 5001:5000| Browser[Browser<br/>localhost:5001]
    
    style API_Image fill:#4CAF50
    style Model_Image fill:#2196F3
    style Flask_App fill:#FF9800
    style Training_Env fill:#9C27B0
```

## 6. Component Interaction Sequence

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Flask_App
    participant Prediction_Engine
    participant Model_Loader
    participant ML_Model
    participant Response
    
    User->>Browser: Fill Form & Submit
    Browser->>Flask_App: POST / (Form Data)
    Flask_App->>Flask_App: Parse Form Data
    Flask_App->>Flask_App: format_categoricals()
    Flask_App->>Prediction_Engine: predict_metrics(data)
    
    Prediction_Engine->>Prediction_Engine: predict([data], 'impressions')
    Prediction_Engine->>Model_Loader: load_from_bucket('impressions_model.pkl')
    Model_Loader->>ML_Model: Load model file
    ML_Model-->>Model_Loader: Model object
    Model_Loader-->>Prediction_Engine: Model object
    
    Prediction_Engine->>Prediction_Engine: Load columns file
    Prediction_Engine->>Prediction_Engine: Reindex features
    Prediction_Engine->>ML_Model: model.predict(features)
    ML_Model-->>Prediction_Engine: Direct prediction
    
    Prediction_Engine->>Prediction_Engine: Calculate CPX
    Prediction_Engine->>Prediction_Engine: Transfer model prediction
    Prediction_Engine-->>Flask_App: Final predictions
    
    Flask_App->>Flask_App: Calculate confidence ranges
    Flask_App->>Response: Render template with results
    Response-->>Browser: HTML with predictions
    Browser-->>User: Display results
```

## 7. Model Types and Evaluation

```mermaid
graph TD
    subgraph "Regression Models"
        Linear[Linear Regression<br/>OLS Method]
        Tree[Decision Tree Regressor<br/>Tree-based]
        Forest[Random Forest Regressor<br/>Ensemble Method]
        SVR[Support Vector Regressor<br/>Kernel-based]
    end
    
    subgraph "Model Selection"
        Train_Acc[Training Accuracy<br/>Mean Relative Accuracy]
        Test_Acc[Test Accuracy<br/>Mean Relative Accuracy]
        Compare[Compare Accuracies]
    end
    
    subgraph "Data Types"
        Scaled[Scaled Data<br/>For SVR]
        Unscaled[Unscaled Data<br/>For Linear/Tree/Forest]
        Categorical[Categorical Data<br/>For CatBoost]
    end
    
    Linear --> Unscaled
    Tree --> Unscaled
    Forest --> Unscaled
    SVR --> Scaled
    
    Linear --> Train_Acc
    Tree --> Train_Acc
    Forest --> Train_Acc
    SVR --> Train_Acc
    
    Linear --> Test_Acc
    Tree --> Test_Acc
    Forest --> Test_Acc
    SVR --> Test_Acc
    
    Train_Acc --> Compare
    Test_Acc --> Compare
    Compare --> Best_Model[Best Model Selected]
    
    style Linear fill:#4CAF50
    style Tree fill:#2196F3
    style Forest fill:#FF9800
    style SVR fill:#9C27B0
    style Best_Model fill:#F44336
```

## 8. Feature Engineering Pipeline

```mermaid
flowchart LR
    Raw_Data[Raw Campaign Data] --> Filter_Valid[Filter Valid Records<br/>output > 0]
    
    Filter_Valid --> Drop_Missing[Drop High-Missing Columns<br/>threshold: 50%]
    Drop_Missing --> Handle_NaN[Drop Rows with NaN]
    Handle_NaN --> Create_Other[Create 'Other' Buckets<br/>threshold: 10%]
    
    Create_Other --> Encode[One-Hot Encoding]
    Encode --> Categorical_Features[Categorical Features<br/>region_*, category_*, shop_*]
    
    Create_Other --> Numerical_Features[Numerical Features<br/>budget, days, weeks, etc.]
    
    Categorical_Features --> Feature_Vector[Final Feature Vector]
    Numerical_Features --> Feature_Vector
    
    Feature_Vector --> Scale[Standard Scaling<br/>For SVR]
    Feature_Vector --> No_Scale[No Scaling<br/>For Tree Models]
    
    Scale --> Scaled_Features[Scaled Features]
    No_Scale --> Unscaled_Features[Unscaled Features]
    
    Scaled_Features --> Model_Input
    Unscaled_Features --> Model_Input[Model Input]
    
    style Raw_Data fill:#4CAF50
    style Feature_Vector fill:#2196F3
    style Model_Input fill:#FF9800
```

## 9. Prediction Request Flow

```mermaid
flowchart TD
    Request[API Request] --> Route{Route Type}
    
    Route -->|GET /| Web_Form[Render Web Form]
    Route -->|POST /| Form_Handler[Handle Form Submission]
    Route -->|GET /ping| Health[Return Health Status]
    Route -->|POST /:metric| Metric_Handler[Single Metric Handler]
    Route -->|POST /campaign| Campaign_Handler[Campaign Handler]
    
    Form_Handler --> Parse_Form[Parse Form Data]
    Parse_Form --> Extract_Fields[Extract Fields:<br/>budget, dates, channels, etc.]
    
    Metric_Handler --> Parse_JSON[Parse JSON Data]
    Campaign_Handler --> Parse_JSON
    
    Extract_Fields --> Format_Data[Format Data]
    Parse_JSON --> Format_Data
    
    Format_Data --> Categorical_Encode[Encode Categoricals]
    Categorical_Encode --> Feature_Extract[Extract Features]
    
    Feature_Extract --> Load_Model[Load Model Files]
    Load_Model --> Align[Align Features]
    Align --> Predict[Run Prediction]
    
    Predict --> Calculate[Calculate Metrics]
    Calculate --> Format_Response[Format Response]
    
    Format_Response -->|Web Form| Render_HTML[Render HTML Template]
    Format_Response -->|API| Return_JSON[Return JSON]
    
    Render_HTML --> Response[HTTP Response]
    Return_JSON --> Response
    Health --> Response
    
    style Request fill:#4CAF50
    style Predict fill:#2196F3
    style Response fill:#FF9800
```

## 10. System Deployment Flow

```mermaid
flowchart TD
    Start([Development]) --> Build_API[Build API Image<br/>Dockerfile.api]
    Start --> Build_Model[Build Model Image<br/>Dockerfile.model]
    
    Build_API --> Install_Deps[Install Dependencies]
    Install_Deps --> Copy_Code[Copy API Code]
    Copy_Code --> Copy_Models[Copy Model Files]
    Copy_Models --> Create_User[Create Non-Root User]
    Create_User --> API_Image[API Image Ready]
    
    Build_Model --> Install_ML_Deps[Install ML Dependencies]
    Install_ML_Deps --> Copy_Model_Code[Copy Model Code]
    Copy_Model_Code --> Model_Image[Model Image Ready]
    
    API_Image --> Docker_Compose[Docker Compose]
    Model_Image --> Docker_Compose
    
    Docker_Compose --> Start_API[Start API Container<br/>Port 5001]
    Docker_Compose --> Ready_Model[Model Container Ready<br/>On-Demand]
    
    Start_API --> Health_Check[Health Check<br/>/ping endpoint]
    Health_Check --> Running[System Running]
    
    Ready_Model --> Train_Command[Run Training Commands]
    Train_Command --> Training[Execute Training]
    Training --> Save_Models[Save New Models]
    Save_Models --> Update_API[Update API Models]
    
    style Start fill:#4CAF50
    style Running fill:#2196F3
    style Training fill:#FF9800
```

## Diagram Legend

- **Green**: Entry/Exit points, successful operations
- **Blue**: Core processing components
- **Orange**: Models and predictions
- **Purple**: Training and evaluation
- **Red**: Critical operations

## Notes

- All diagrams use Mermaid syntax and can be rendered in:
  - GitHub/GitLab markdown
  - VS Code with Mermaid extension
  - Online Mermaid editors
  - Documentation tools that support Mermaid

- The system supports both web interface and REST API access
- Models are trained offline and loaded at runtime
- Docker containers provide isolated environments for API and training
- Model files are shared between containers via volume mounts

