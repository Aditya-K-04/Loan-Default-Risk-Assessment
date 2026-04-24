Company : CodeTech IT Solutions
Name : Aditya Kurkote
Intern ID : CTIS4778
Domain : Data Science Intern
Duration : 12 weeks
Mentor : Neela Santosh Kumar


# Loan-Default-Risk-Assessment
Create folders.

Install packages.

Run prepare_data.py to make loan_data.csv.

Copy loan_data.csv into Task 1, 2, and 3.

Run Task 1 script.

Run Task 2 script.

Run Task 3 training script, then start FastAPI.

Create and run Task 4 notebook.

This project focuses on building a machine learning-based system to assess the risk of loan default for credit applicants. In the financial sector, banks and lending companies must make accurate decisions about whether to approve a loan or reject it. If a risky applicant is approved, the lender may face financial loss. If a good applicant is rejected, the company may lose a valuable customer. To reduce this risk, the project aims to develop a predictive model that can identify whether a loan applicant is likely to default.

The project uses the UCI Credit Approval dataset, which is a simple and widely used dataset for credit-related classification tasks. It contains applicant information such as personal details, financial attributes, and credit-related factors. Since the dataset includes both numerical and categorical values, it is suitable for demonstrating data preprocessing, feature transformation, model training, and optimization. The target variable indicates whether the applicant is approved or rejected, which is used to build a classification model for risk assessment.

The first task in this project is data preprocessing and pipeline creation. In this step, missing values are handled, categorical features are encoded, and numerical features are scaled. A complete preprocessing pipeline is created using Pandas and scikit-learn so that the data can be cleaned and transformed automatically. This step is important because machine learning models perform better when the input data is properly prepared.

The second task is deep learning model development. A neural network model is built using TensorFlow to classify applicants based on their risk level. The model is trained on the processed credit data and evaluated using metrics such as accuracy, AUC score, and classification report. Training curves are also visualized to show how the model learns over time. This task demonstrates how deep learning can be applied to structured financial data.

The third task is end-to-end deployment. In this part, the trained model is integrated into a FastAPI application. The user can send applicant details to the API and receive a prediction about the default risk. This makes the project practical and usable in a real-world scenario. The API can be tested locally and later deployed on a cloud platform such as Render.

The fourth task is optimization using PuLP. In this task, the business problem is framed as a loan approval optimization problem. Based on loan amount, expected profit, and default probability, the model selects the best set of applicants while keeping risk under a fixed threshold. This helps demonstrate how optimization techniques can support financial decision-making.

Overall, this project combines data preprocessing, machine learning, deep learning, API deployment, and optimization in one complete workflow. It is a strong internship project because it shows both technical skills and business understanding. It also follows a real-world use case that is highly relevant to banking, fintech, and credit-risk analysis.
