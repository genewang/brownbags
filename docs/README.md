# System Documentation

This directory contains comprehensive system diagrams and architecture documentation for the Marketing Performance Predictor project.

## Documentation Files

### 1. [system-diagrams.md](./system-diagrams.md)
Comprehensive collection of 10 detailed system diagrams including:
- System Architecture Overview
- Data Flow Diagram
- API Endpoints Structure
- Model Training Pipeline
- Docker Deployment Architecture
- Component Interaction Sequence
- Model Types and Evaluation
- Feature Engineering Pipeline
- Prediction Request Flow
- System Deployment Flow

### 2. [architecture-overview.md](./architecture-overview.md)
High-level architecture overview with:
- Visual ASCII architecture diagrams
- Component relationships
- Data flow summary
- Technology stack

## Viewing the Diagrams

The diagrams are written in **Mermaid** syntax, which can be rendered in:

1. **GitHub/GitLab**: Automatically rendered in markdown files
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live/)
4. **Documentation Tools**: Most modern documentation tools support Mermaid

## Quick Reference

### System Components
- **API Service**: Flask web application serving predictions (Port 5001)
- **Model Training**: Offline training environment for ML models
- **Model Storage**: Pickle files (.pkl) containing trained models and feature schemas
- **Docker Containers**: Isolated environments for API and training

### Key Endpoints
- `GET /` - Web interface for campaign predictions
- `GET /ping` - Health check endpoint
- `POST /:metric` - Single metric prediction (JSON)
- `POST /campaign` - Full campaign prediction (JSON)

### Model Types
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)

## Diagram Types

1. **Architecture Diagrams**: Show system structure and component relationships
2. **Flow Diagrams**: Illustrate data and process flows
3. **Sequence Diagrams**: Show interaction sequences between components
4. **Deployment Diagrams**: Show Docker container architecture

## Contributing

When updating diagrams:
1. Use Mermaid syntax for consistency
2. Follow the color scheme:
   - Green: Entry/Exit points
   - Blue: Core processing
   - Orange: Models/Predictions
   - Purple: Training/Evaluation
   - Red: Critical operations
3. Keep diagrams focused and readable
4. Update this README if adding new diagram files

