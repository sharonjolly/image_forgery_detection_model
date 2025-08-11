#Goal
This project focuses on creating, monitoring, and deploying an Image Forgery Detection system trained on the CASIA2 dataset. It leverages a custom CNN architecture tailored for precise feature extraction and robust classification. The pipeline is fully integrated with MLflow for experiment tracking, DVC for dataset and workflow management, and an automated CI/CD process that deploys to AWS via Docker and GitHub Actions.

1.Project Setup and Configuration
Centralized configuration:
  config.yaml — stores all directory paths and global settings.
  params.yaml — defines preprocessing and training hyperparameters.
Structured config management:
  Created entity classes for strongly-typed configuration handling.
  Implemented a Configuration Manager (src/config) to load and manage settings.
Modular components for:
  Data ingestion
  Data preprocessing
  Model training
  Model evaluation
Automated pipeline:
  Linked all components into a cohesive, automated workflow.
  Added main.py to trigger pipelines.

Updated dvc.yaml to define and track each DVC pipeline stage.

