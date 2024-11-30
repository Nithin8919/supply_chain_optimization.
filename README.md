# Supply Chain Optimization 

## **Project Overview**
This project is designed to address supply chain optimization for a well-known retail company in the USA. The primary objective is to predict the **`product_wg_ton`**, which refers to the weight in tons of products that should be available to meet customer demand efficiently.

Accurate demand forecasting ensures:
- Optimal inventory levels.
- Prevention of stockouts or overstocking.
- Enhanced customer satisfaction.
- Cost savings through efficient resource allocation.

This project integrates **Machine Learning (ML)** models with robust tools and frameworks for seamless experimentation, deployment, and monitoring.

---

## **Key Features**
1. **Demand Forecasting**: Predicting `product_wg_ton` for different product categories using historical sales data and external influencing factors.
2. **Pipeline Automation**: The use of tools like **Airflow** to automate data processing, training, and deployment pipelines.
3. **Experiment Tracking**: Employing **MLflow** for tracking model performance and maintaining reproducibility.
4. **Deployment**: Seamlessly deploying trained models via **MLflow Serving** .
5. **Version Control**: Utilizing **DVC** to manage large datasets and their versions, ensuring data consistency across experiments.

---

## **Architecture**

![Project Architecture](path_to_image/project_architecture.png)

The project follows a modular architecture:
1. **Data Ingestion**:
   - Collect data from multiple sources (e.g., sales, inventory, external factors like holidays or weather).
   - Store data in **MongoDB** for structured querying.

2. **Exploratory Data Analysis (EDA) & Visualization**:
   - Understand data trends and patterns using tools like **Matplotlib** and **Seaborn** and **Plotly**.

3. **Data Cleaning**:
   - Remove inconsistencies, handle missing values, and ensure data quality.

4. **Feature Engineering**:
   - Generate features (e.g., seasonality, lagged variables) to enhance model performance.

5. **Model Training and Evaluation**:
   - Train ML models (e.g., Random Forest, XGBoost) to predict demand (`product_wg_ton`).
   - Use **MLflow** to log metrics, artifacts, and model versions.

6. **Model Deployment**:
   - Deploy the model via **MLflow Serving** and expose it as a REST API.

7. **Monitoring**:
   - Use **Grafana** dashboards to track the performance of deployed models and monitor predictions in real time.

---

## **Technologies Used**

### **Languages**
- Python

### **Tools and Frameworks**
- **MLflow**: Experiment tracking and model deployment.
- **DVC (Data Version Control)**: For dataset versioning and consistency.
- **Airflow**: Pipeline orchestration.
- **Github Actions**: CI/CD Pipelines.
- **MongoDB**: Data storage.

---

## **Dataset**

The dataset includes:
- **Historical Sales Data**: Features like date, product category, and sales volume.
- **External Factors**: Holidays, promotions, and weather information.

## **Project Orchestration**
- **Airflow DAGs**: Define and manage workflows for data ingestion, EDA, model training
![Airflow Dags]![IMG_20241130_113548](https://github.com/user-attachments/assets/40acdbcb-dac1-4f34-ae54-fd9611df8da6)

- **MLFlow Evaluation**:
- 

## **How to Run the Project**

### **Prerequisites**
- Python >= 3.8
- Docker (optional for containerized deployment)

### **Steps**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/supply-chain-optimization.git
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
   python train.py
   ```

5. **Serve the Model**
   ```bash
   mlflow models serve --model-uri models:/SupplyChainModel/Production --port 5000
   ```

6. **Run the Streamlit App**
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
- **Your Name**  
  [GitHub](https://github.com/your_username) | [LinkedIn](https://www.linkedin.com/in/your_profile)

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

Let me know if you’d like further adjustments or more details added!
