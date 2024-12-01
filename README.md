# Supply Chain Optimization

## **Project Overview**
This project addresses supply chain optimization for a well-known retail company in the USA. The primary objective is to predict **`product_wg_ton`**, which refers to the weight in tons of products that should be available to meet customer demand efficiently.

Accurate demand forecasting ensures:
- Optimal inventory levels.
- Prevention of stockouts or overstocking.
- Enhanced customer satisfaction.
- Cost savings through efficient resource allocation.

This project integrates **Machine Learning (ML)** models with robust tools and frameworks for seamless experimentation, deployment, and monitoring, while leveraging **AWS** services for efficient cloud-based deployment and storage.

---

## **Key Features**
1. **Demand Forecasting**: Predicting `product_wg_ton` for different product categories using historical sales data and external influencing factors.
2. **Pipeline Automation**: The use of tools like **Airflow** to automate data processing, training, and deployment pipelines.
3. **Experiment Tracking**: Employing **MLflow** for tracking model performance and maintaining reproducibility.
4. **Cloud Deployment**: 
   - **Dockerized** model storage and execution for portability.
   - Continuous Integration (CI), Continuous Delivery (CD), and Continuous Deployment (CD) pipelines using **GitHub Actions** integrated with **AWS S3**, **ECR**, and **EC2**.
5. **Version Control**: Utilizing **DVC** to manage large datasets and their versions, ensuring data consistency across experiments.

---

## **Architecture**

![SCO Architecture](https://github.com/user-attachments/assets/5b54786d-6455-4f62-9c4e-34af53219fcd)

The project follows a modular architecture:
1. **Data Ingestion**:
   - Collect data from multiple sources (e.g., sales, inventory, external factors like holidays or weather).
   - Store data in **MongoDB** for structured querying.

2. **Exploratory Data Analysis (EDA) & Visualization**:
   - Understand data trends and patterns using tools like **Matplotlib**, **Seaborn**, and **Plotly**.

3. **Data Cleaning**:
   - Remove inconsistencies, handle missing values, and ensure data quality.

4. **Feature Engineering**:
   - Generate features and added synthetic data(e.g., seasonality) to enhance model performance.

5. **Model Training and Evaluation**:
   - Train ML models (e.g., Random Forest, XGBoost) to predict demand (`product_wg_ton`).
   - Use **MLflow** to log metrics, artifacts, and model versions.

6. **Cloud Deployment**:
   - Save models as Docker images and push to **AWS ECR**.
   - Store artifacts in **AWS S3** and deploy on **AWS EC2** instances.

---

## **Technologies Used**

### **Languages**
- Python

### **Tools and Frameworks**
- **MLflow**: Experiment tracking and model deployment.
- **DVC (Data Version Control)**: For dataset versioning and consistency.
- **Airflow**: Pipeline orchestration.
- **GitHub Actions**: CI/CD Pipelines.
- **AWS**: S3 for storage, ECR for Docker image registry, EC2 for cloud deployment.
- **Docker**: Containerization for model portability.
- **MongoDB**: Data storage.

---

## **Dataset**

The dataset includes:
- **Historical Sales Data**: Features like date, product category, and sales volume.
- **External Factors**: Holidays, promotions, and weather information.

---

## **Project Orchestration**

### **Airflow DAGs**
Manage workflows for data ingestion, EDA, and model training.
![airflow img](https://github.com/user-attachments/assets/8fffd44d-d273-425a-a9d0-bb6d78781ea1)

### **MLflow Evaluation**
Track metrics and artifacts.
![MLflow Evaluation](https://github.com/user-attachments/assets/05387458-7709-4b95-8fd3-b6e14b96e3d3)

### **CI/CD/CD Pipeline**
- Dockerized models stored in **AWS ECR**.
- Artifacts stored in **AWS S3**.
- Deployment on **AWS EC2** instances using **GitHub Actions**.
![CI/CD/CD](https://github.com/user-attachments/assets/614de2ed-fbd3-4349-898f-4a813d69496b)

---

## **How to Run the Project**

### **Prerequisites**
- Python >= 3.8
- Docker
- AWS CLI configured with proper permissions.

### **Steps**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nithin8919/supply-chain-optimization.git
   cd supply-chain-optimization
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Airflow**
   - Initialize Airflow:
     ```bash
     airflow db init
     airflow scheduler
     airflow webserver
     ```
   - Configure DAGs for automated pipeline execution.

4. **Train the Model**
   ```bash
   python training_pipeline.py
   ```

5. **Dockerize and Push the Model**
   - Build Docker image:
     ```bash
     docker build -t supply_chain_model .
     ```
   - Push to AWS ECR:
     ```bash
     aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
     docker tag supply_chain_model:latest <account_id>.dkr.ecr.<region>.amazonaws.com/supply_chain_model:latest
     docker push <account_id>.dkr.ecr.<region>.amazonaws.com/supply_chain_model:latest
     ```

6. **Serve the Model**
   - Deploy using EC2:
     ```bash
     docker run -d -p 5000:5000 <account_id>.dkr.ecr.<region>.amazonaws.com/supply_chain_model:latest
     ```

7. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

---

## **Results**
- The model predicts the required `product_wg_ton` with high accuracy, ensuring optimal inventory levels.
- Significant reduction in supply chain costs by balancing demand and inventory.

---

## **Future Work**
1. Expand to multi-region predictions for scalability.
2. Incorporate real-time data for dynamic forecasting.
3. Enhance model interpretability with SHAP or LIME.

---

## **Contributors**
- **Nithin CH**  
  [GitHub](https://github.com/Nithin8919) | [LinkedIn](https://www.linkedin.com/in/nithin-ch-7a478b21a/)

---

## **License**
This project is licensed under the [MIT License](LICENSE).
