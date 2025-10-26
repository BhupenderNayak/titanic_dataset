# 🚢 Titanic Survival Prediction: My First Deep Learning Project



<p align="center">
  <img src="https://img.shields.io/badge/Kaggle%20Score-84.27%25-blue.svg" alt="Kaggle Score">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Used-red.svg" alt="Keras">
</p>

This repository documents my first end-to-end machine learning project, built for the classic **Kaggle "Titanic - Machine Learning from Disaster" competition**.

The primary goal was to process and clean a real-world, "messy" dataset and then build a deep learning (neural network) model from scratch to predict passenger survival.

**My final submission achieved a 84.27% accuracy on the unseen test set.**

---

## 🛠️ Technologies & Libraries

* **Data Analysis & Cleaning:** Pandas, NumPy
* **Data Preprocessing:** Scikit-learn (`StandardScaler`, `train_test_split`)
* **Deep Learning Framework:** TensorFlow with the Keras API
* **Visualization:** Matplotlib & Pandas Plotting

---

## 📂 Project Workflow

I followed the complete data science pipeline from data ingestion to model training and submission.

### 1. Data Cleaning & Preprocessing

The raw dataset was incomplete and contained non-numeric data that a neural network cannot process.

* **Missing `Age`:** Filled 177 missing `Age` values with the dataset's **median age**.
* **Missing `Fare`:** Filled one missing `Fare` value with the dataset's **median fare**.
* **Missing `Embarked`:** Filled two missing `Embarked` values with the **mode** ('S'), which is the most common port.
* **Dropped `Cabin`:** This column was missing over 77% of its data, so it was dropped entirely.
* **Dropped `Name` & `Ticket`:** These were unique identifiers with no general predictive pattern, so they were removed.

### 2. Feature Engineering

To convert the cleaned data into a purely numerical format for the model, I engineered the following features:

* **`Sex`:** Converted from (`male`, `female`) to a single one-hot encoded column, `Sex_male` (1 or 0).
* **`Embarked`:** Converted from (`S`, `C`, `Q`) to two one-hot encoded columns, `Embarked_Q` and `Embarked_S`.
* **`StandardScaler`:** All numerical features (e.g., `Age`, `Fare`, `Pclass`) were scaled to have a mean of 0 and a standard deviation of 1. This is critical for helping the neural network train efficiently.

### 3. Deep Learning Model Architecture

I built a `Sequential` model in Keras designed to be robust against overfitting, which is a common problem with small datasets like this one.

Here is the final model summary:

| Layer Type | Output Shape | Parameters | Purpose |
| :--- | :--- | :--- | :--- |
| `Dense` | (None, 128) | 1,152 | First hidden "thinking" layer |
| `Dropout` | (None, 128) | 0 | Prevents overfitting (rate: 0.3) |
| `BatchNormalization` | (None, 128) | 512 | Stabilizes training |
| `Dense` | (None, 64) | 8,256 | Second hidden "thinking" layer |
| `Dropout` | (None, 64) | 0 | Prevents overfitting (rate: 0.3) |
| `BatchNormalization` | (None, 64) | 256 | Stabilizes training |
| `Dense` (Output) | (None, 1) | 65 | **Sigmoid** activation; outputs survival probability (0.0 to 1.0) |

### 4. Training & Evaluation

* **Optimizer:** `adam` (a fast and efficient default)
* **Loss Function:** `binary_crossentropy` (the standard for yes/no classification)
* **Callback: `EarlyStopping`**
    * This was the most important part of training. I set it to monitor `val_loss` (the "practice exam" score).
    * The model automatically stopped training when the validation score stopped improving, which prevented overfitting and saved the best version of the model.

The model trained quickly and showed stable validation accuracy, proving the `Dropout` and `BatchNormalization` layers were effective.



---

## 🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib
    ```

3.  **Run the notebook:**
    * Open and run the `.ipynb` notebook (e.g., `titanic_model.ipynb`) in Jupyter Notebook or Google Colab.
    * The notebook will load the data, clean it, train the model, and generate the `submission.csv` file.

---

## 🎓 Key Learnings

* **Feature Engineering is King:** A model is only as good as its data. Cleaning, filling, and encoding data (like `Age` and `Sex`) is the most important part of the process.
* **Preventing Overfitting is Critical:** With a small dataset, a deep learning model will memorize it very quickly. Using `Dropout` and `EarlyStopping` was essential to building a model that could generalize to new data.
* **Data Scaling Matters:** `StandardScaler` made a noticeable difference in how quickly and stably the model trained.
