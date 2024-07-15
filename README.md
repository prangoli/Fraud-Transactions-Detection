# Credit-Card Fraud Detection System using Machine Learning

## Introduction
This project focuses on detecting fraudulent credit card transactions using a neural network with entity embeddings for categorical variables. The dataset spans transactions from January 1, 2019, to December 31, 2020, and includes over 1.5 million transactions from 953 unique customers. The imbalanced nature of the dataset, where less than 0.1% of the transactions are fraudulent, presents significant challenges and opportunities for advanced machine learning techniques.

## Dataset
### About the Dataset
This simulated dataset contains both legitimate and fraudulent transactions from the period January 1, 2019, to December 31, 2020. It covers credit cards of 953 customers performing transactions with a pool of 800 merchants. The data was generated using the Sparkov Data Generation tool created by Brandon Harris, simulating realistic transaction scenarios.

### Source of Simulation
The dataset was generated using the Sparkov Data Generation tool. This simulation uses predefined lists of merchants, customers, and transaction categories, leveraging the "faker" library to create realistic transaction data. Various customer profiles, such as age groups and regions, were considered to ensure a diverse dataset. These profiles determine transaction patterns, including the frequency and amount of transactions.

### Information about the Simulator
The simulator uses a predefined list of merchants, customers, and transaction categories. Using the "faker" library, transactions are generated based on these lists. Depending on the chosen customer profile (e.g., "adults 25-50 female rural"), transactions are created with specific properties such as transaction frequency, distribution across days of the week, and normal distribution properties for amounts in various categories. These simulated transactions are merged to create a realistic representation of credit card usage.

### Original Columns
- **index:** Unique Identifier for each row
- **trans_date_trans_time:** Transaction DateTime
- **cc_num:** Credit Card Number of Customer
- **merchant:** Merchant Name
- **category:** Category of Merchant
- **amt:** Amount of Transaction
- **first:** First Name of Credit Card Holder
- **last:** Last Name of Credit Card Holder
- **gender:** Gender of Credit Card Holder
- **street:** Street Address of Credit Card Holder
- **city:** City of Credit Card Holder
- **state:** State of Credit Card Holder
- **zip:** Zip of Credit Card Holder
- **lat:** Latitude Location of Credit Card Holder
- **long:** Longitude Location of Credit Card Holder
- **city_pop:** Credit Card Holder's City Population
- **job:** Job of Credit Card Holder
- **dob:** Date of Birth of Credit Card Holder
- **trans_num:** Transaction Number
- **unix_time:** UNIX Time of transaction
- **merch_lat:** Latitude Location of Merchant
- **merch_long:** Longitude Location of Merchant
- **is_fraud:** Fraud Flag (Target Class)

### Selected Columns for Model
- **Categorical Variables:** `merchant`, `category`, `gender`, `city`, `state`, `job`
- **Continuous Variables:** `amt`, `lat`, `long`, `city_pop`, `merch_lat`, `merch_long`
- **Target:** `is_fraud`

### Post-Processing Columns
- **mean_monthly_amt**
- **amount_of_monthly_trans**
- **age**
- **mean_time_between_transactions_seconds**

## Problem Statement
The objective of this project is to identify fraudulent credit card transactions. The primary challenge is the imbalanced nature of the dataset, where fraudulent transactions are less than 0.1%. Traditional rule-based systems are inadequate for such a task, necessitating the use of advanced machine learning techniques.

## Model
The project employs a neural network with entity embeddings for categorical variables. The model leverages several key techniques:

### Key Techniques Used:
- **Neural Networks:** These models are capable of capturing complex patterns in data, making them suitable for fraud detection.
- **Entity Embeddings:** These embeddings transform categorical variables into continuous vectors, enabling the neural network to process them more efficiently. This method helps in capturing the relationships between different categorical variables.
- **Dropout:** This regularization technique helps prevent overfitting by randomly dropping a fraction of the input units during training. It forces the network to learn more robust features that generalize well to new data.

### Data Preprocessing
1. **Categorical Variables:** The categorical variables such as `merchant`, `category`, `gender`, `city`, `state`, and `job` are encoded using entity embeddings.
2. **Continuous Variables:** Continuous variables including `amt`, `lat`, `long`, `city_pop`, `merch_lat`, and `merch_long` are normalized to ensure they are on a comparable scale.
3. **Feature Engineering:** New features such as `mean_monthly_amt`, `amount_of_monthly_trans`, `age`, and `mean_time_between_transactions_seconds` are created to enhance the model's predictive power.

### Model Architecture
The neural network architecture includes:
- **Input Layer:** Accepts both continuous and embedded categorical variables.
- **Hidden Layers:** Multiple layers with ReLU activation functions.
- **Dropout Layers:** Applied after hidden layers to prevent overfitting.
- **Output Layer:** A single neuron with a sigmoid activation function to predict the probability of a transaction being fraudulent.

## Results
![Model Results](path/to/results/image.png)

## Acknowledgements
- Thanks to [Kartik2112](https://www.kaggle.com/kartik2112) for providing the [dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraudTrain.csv).
- Thanks to Brandon Harris for creating the [Sparkov Data Generation tool](https://github.com/brandonharris/sparkov-data-generation).




