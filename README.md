# 🛒 E-Commerce Purchase Prediction

## 📌 Problem Statement

An e-commerce company wants to predict whether a visitor will make a purchase based on their browsing session behavior.
The goal is to build a Machine Learning model that can classify sessions into **purchase (Revenue = True)** or **no purchase (Revenue = False)**.

---

## 📊 Dataset

* ~12,330 user sessions
* Mix of numerical and categorical features
* Target variable: **Revenue**

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)

* Checked data distribution
* Identified class imbalance
* Visualized key features

### 2. Data Preprocessing

* Handled categorical variables
* Feature transformations
* Prepared dataset for model training

### 3. Model Building

* Decision Tree Classifier
* Applied pruning to avoid overfitting
* Handled class imbalance using class weights
* Hyperparameter tuning using **GridSearchCV**

### 4. Evaluation

* Metric used: **F1 Score** (due to imbalanced dataset)

---

## 🌳 Model Details

* Algorithm: Decision Tree Classifier
* Techniques used:

  * Pruning (`max_depth`, `min_samples_split`)
  * Class balancing
  * Hyperparameter tuning (GridSearchCV)

---

## 📈 Results

* Achieved F1 Score: **~0.55+** (target met)

---

## 📂 Project Structure

```bash
ECOMMERCE-PURCHASE-PREDICTION/
│
├── data/
│   └── raw/
│        └── shop_smart_ecommerce.csv   # Original dataset
│
├── models/
│   └── best_model.pkl           # Saved trained model
│
├── notebooks/
│    └── Ecommerce_P...ipynb      # Main notebook (EDA + training)
│
├── src/
│   ├── evaluate.py              # Model evaluation (F1, metrics)
│   ├── preprocess.py            # Data preprocessing
│   ├── train.py                 # Model training
│   └── utils.py                 # Helper functions
│
├── .gitignore
├── main.py                      # Entry point script
├── README.md
├── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ecommerce-purchase-prediction.git
cd ecommerce-purchase-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## 📌 Author

Abhay Singh
