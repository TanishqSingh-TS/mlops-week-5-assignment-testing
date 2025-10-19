# MLOps Assignment: CI/CD Pipeline for Model Validation and Evaluation using CML

This repository contains the files for an MLOps assignment focused on building an automated CI/CD pipeline using GitHub Actions. The pipeline is designed to validate incoming data and evaluate a classification model's performance automatically.

---

## 📂 Repository Structure

Here's an overview of the key files and directories in this project:

* **`.github/workflows/main.yml`**: Contains the core CI/CD pipeline logic.
* **`test_data/`**: Includes data validation tests using `pytest`.
* **`check_metrics/`**: A module to generate and report model performance metrics.
