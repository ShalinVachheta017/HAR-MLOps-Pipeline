# 📋 Thesis Plan

> **📝 Summary:** This document outlines the complete 6-month thesis timeline (Oct 2025 - Apr 2026) broken down by month and week. It covers: Month 1 (Data Ingestion), Month 2 (Model Training & Versioning), Month 3 (CI/CD & Deployment), Month 4 (Monitoring & Integration), Month 5 (Refinement & Retraining), Month 6 (Thesis Writing). Use this as the master roadmap for thesis milestones.

---

**Thesis Title:** Developing a MLOps Pipeline for Continuous Mental Health Monitoring using Wearable Sensor Data

**Overall Goal for 6 Months:** To establish a foundational MLOps pipeline for anxiety activity recognition and prognosis models, demonstrating automated data handling, model management, and basic monitoring. The emphasis will be on *proof-of-concept* and *scalability principles*, rather than a full production-grade system.

**Phase Breakdown (Approximate Timeline):**

- **Month 1: Research, Planning & Data Ingestion Setup (Focus on Data)**
    - **Week 1-2: Literature Review & Detailed Planning**
        - Deep dive into existing MLOps best practices, especially for time series data and healthcare applications.
        - Familiarize with the current anxiety activity recognition model (1D-CNN-BiLSTM) and the data structure (Garmin accelerometer/gyroscope).
        - Define exact scope and success metrics for the 6 months.
    - **Week 3-4: Data Ingestion & Preprocessing Pipeline (Initial)**
        - Set up a robust system for ingesting raw Garmin data (or simulated continuous streams if real-time access is limited).
        - Implement initial preprocessing steps (e.g., handling missing values, resampling, feature engineering relevant to the 1D-CNN-BiLSTM).
        - Focus on making this process automated and reproducible using Python scripts and Pandas/NumPy.
- **Month 2: Model Training & Versioning (Focus on Reproducibility)**
    - **Week 5-6: Automated Training Loop Integration**
        - Integrate your existing anxiety activity recognition model into an automated training script.
        - Implement logging (e.g., using MLflow or a custom logger) for metrics and parameters.
        - Integrate automated hyperparameter optimization within the training loop.
    - **Week 7-8: Model Versioning & Experiment Tracking**
        - Implement model versioning (e.g., using MLflow Model Registry,or a structured file system).
        - Ensure that every training run is trackable and reproducible. Containerize the training environment using Docker.
- **Month 3: CI/CD (Continuous Integration/Continuous Deployment) for Models & Basic Deployment (Focus on Automation)**
    - **Week 9-10: CI/CD Pipeline Setup**
        - Set up a basic CI/CD pipeline using GitHub Actions  for the activity recognition model.
        - This should include automated testing of code, building Docker images, and potentially pushing them to a registry.
    - **Week 11-12: Model Serving & Basic API Deployment**
        - Develop a simple API (e.g., using Flask or FastAPI ) to serve the trained activity recognition model for inference.
        - Containerize the inference service with Docker  for reproducible deployment.
        - This serves as the "online" component where new data streams could be processed.
- **Month 4: Integration with Prognosis Model & Initial Monitoring (Focus on Flow)**
    - **Week 13-14: Activity to Prognosis Data Flow**
        - Design and implement the data flow from the recognized activities (output of the first model) as input to your second prognosis model.
        - This might involve storing intermediate results in a simple database or file system.
    - **Week 15-16: Basic Model Monitoring**
        - Implement basic monitoring for the *deployed* activity recognition model:
            - **Data Drift:** Monitor incoming sensor data characteristics (e.g., mean, variance) to detect significant changes.
            - **Prediction Drift:** Monitor the distribution of the model's output predictions.
            - **Performance Monitoring:** If ground truth labels become available eventually, monitor accuracy/F1 score.
        - Set up simple alerts or visualizations (Matplotlib, Seaborn ) for these metrics.
- **Month 5: Refinement, Documentation & Initial Retraining Strategy (Focus on Improvement)**
    - **Week 17-18: Pipeline Refinement & Robustness**
        - Address any bottlenecks or instabilities identified in the pipeline.
        - Improve error handling and logging across all components.
    - **Week 19-20: Initial Retraining Strategy (Concept & Prototype)**
        - Develop a conceptual design for an automated or semi-automated retraining trigger based on the monitoring metrics.
        - Implement a *prototype* of this retraining trigger, demonstrating how the pipeline would initiate a new training run if drift is detected. This might not involve full retraining but a demonstration of the trigger.
- **Month 6: Thesis Writing & Final Deliverables (Focus on Communication)**
    - **Week 21-24: Thesis Writing & Presentation**
        - Dedicate significant time to writing the thesis document, outlining the problem, methodology, implementation details, results, and future work.
        - Prepare for the final presentation.
        - Ensure all code is well-documented and organized in a GitHub repository.