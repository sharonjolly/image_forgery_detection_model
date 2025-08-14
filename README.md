# IMAGE FORGERY DETECTION MODEL

 ![CICD Pipeline Architecture](https://github.com/user-attachments/assets/9d205f60-9601-4d00-bf31-60f2a242616e)


## Goal

This project focuses on creating, monitoring, and deploying an Image Forgery Detection system trained on the CASIA2 dataset. It leverages a custom CNN architecture tailored for precise feature extraction and robust classification. The pipeline is fully integrated with MLflow for experiment tracking, DVC for dataset and workflow management, and an automated CI/CD process that deploys to AWS via Docker and GitHub Actions.

## 1. Project Setup and Configuration
#### Centralized configuration:
  `config.yaml` — stores all directory paths and global settings.
  `params.yaml` — defines preprocessing and training hyperparameters.
#### Structured config management:
  Created entity classes for strongly-typed configuration handling.
  Implemented a Configuration Manager (src/config) to load and manage settings.
#### Modular components for:
  Data ingestion
  Data preprocessing
  Model training
  Model evaluation
#### Automated pipeline:
  Linked all components into a cohesive, automated workflow.
  Added `main.py` to trigger pipelines.
  Updated `dvc.yaml` to define and track each DVC pipeline stage.

## 2. Model Development
  A convolutional neural network (CNN) architecture was designed and implemented specifically for the task of image       forgery detection. The model was trained and validated using the CASIA2 dataset to ensure robust learning of            discriminative features. Model performance was quantitatively evaluated using multiple metrics, including               accuracy, precision, recall, and F1-score, providing a comprehensive assessment of classification effectiveness.

## 3. Experiment Tracking with MLflow
  MLflow was integrated into the development pipeline to facilitate systematic experiment tracking. The framework was     employed to log model parameters and performance metrics, store trained image forgery detection model artifacts, and    assign version tags for model management. DagsHub was configured as the remote MLflow tracking server, enabling         centralized storage and access to experimental data. The `mlflow ui` interface was utilized for visualizing and         comparing experiments, while environment variables were set to ensure that all experiment logs were automatically       pushed to the remote server.

## 4. Data Version Control with DVC
  Data Version Control (DVC) was initialized to manage both datasets and pipeline stages, ensuring full reproducibility   of experiments through systematic versioning of all data and trained models. The dvc repro command was utilized to      regenerate the pipeline from any specified stage, while `dvc dag` was employed to visualize the workflow structure.     This approach streamlined collaboration, facilitated experiment traceability, and maintained a transparent history of   all experimental iterations.

## 5. Deployment with AWS and GitHub Actions
  An automated continuous integration and continuous deployment (CI/CD) workflow was implemented using GitHub Actions    in conjunction with AWS services. An Amazon EC2 instance was configured as a self-hosted GitHub Actions runner to       execute deployment tasks. Within the pipeline, a Docker image containing the model inference service was built and      subsequently pushed to Amazon Elastic Container Registry (ECR). The EC2 instance then pulled the image from ECR and     deployed it as a containerized service to handle prediction requests. Appropriate AWS Identity and Access Management    (IAM) policies, including AmazonEC2FullAccess and AmazonEC2ContainerRegistryFullAccess, were applied to enable secure   operations. AWS credentials and configuration details were stored as GitHub Secrets to maintain security and            confidentiality during the deployment process.

## 6. Web Application for Image Forgery Detection
A Flask-based web application was developed to provide an interactive interface for end-users. The application         enables users to upload images, which are then analyzed by the trained image forgery detection model to determine       whether the image is forged or authentic. Model inference is facilitated through a RESTful API, with results rendered   via an HTML-based front-end. During development, the application was executed locally using `app.py`, while in the production environment—following CI/CD deployment—it was configured to run on a designated custom port within the AWS infrastructure.


<img width="1911" height="965" alt="Screenshot 2025-08-11 204023" src="https://github.com/user-attachments/assets/a2735889-01ce-4fe4-a674-5a7d197438c3" />

## Outcome
The project resulted in a production-ready image forgery detection system built upon a custom convolutional neural network (CNN) architecture. Automated pipelines for model training and evaluation were implemented using Data Version Control (DVC), ensuring reproducibility and efficient workflow management. Experiment tracking and model versioning were facilitated through MLflow, enabling systematic performance monitoring and artifact management. A fully automated deployment pipeline, leveraging Docker and GitHub Actions, was integrated with AWS services to streamline the transition from development to production. Additionally, a user-friendly web application was developed to enable real-time image forgery detection, supporting PNG, JPEG, and JPG image formats.

## 8. Key Tools and Technologies
The implementation of this project leveraged a range of tools and frameworks to support data processing, model development, experiment tracking, deployment, and user interaction:

* Python — for data preprocessing, model training, and overall workflow orchestration.
* TensorFlow/Keras — to design, train, and evaluate the custom CNN architecture.
* MLflow — for systematic experiment tracking, performance logging, and model versioning.
* DVC — to enable dataset and pipeline version control, ensuring reproducibility.
* Docker — for containerizing the model inference service and its dependencies.
* AWS ECR & EC2 — to host and run the deployed application in a scalable cloud environment.
* GitHub Actions — for automated CI/CD pipelines.
* DagsHub — as the remote MLflow tracking server for centralized experiment management.
*Flask — to develop the web application interface for real-time image forgery detection.

