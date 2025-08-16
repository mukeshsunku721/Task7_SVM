# Task 7: Support Vector Machines (SVM)

## 🎯 Objective
Use Support Vector Machines (SVM) for **linear** and **non-linear classification** on the Breast Cancer dataset.

---

## 🛠️ Tools & Libraries
- Python 3.x  
- Scikit-learn  
- NumPy  
- Matplotlib  

---

## 📂 Dataset
We used the **Breast Cancer Dataset** from Kaggle:  
[Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

---

## 📌 Steps Performed
1. **Load and Prepare Data**  
   - Read dataset using Pandas.  
   - Selected features and labels (`diagnosis`).  
   - Converted categorical labels (`M` = Malignant, `B` = Benign) into numeric values.  
   - Split dataset into **train (80%)** and **test (20%)** sets.

2. **Train Models**
   - Trained SVM with **Linear Kernel**.  
   - Trained SVM with **RBF Kernel** (non-linear).  

3. **Visualization**
   - Reduced dataset to **2D features** for visualization.  
   - Plotted decision boundaries for both kernels.

4. **Hyperparameter Tuning**
   - Used **GridSearchCV** to tune parameters `C` and `gamma`.  
   - Best model parameters were selected automatically.

5. **Model Evaluation**
   - Evaluated using **Confusion Matrix** and **Classification Report**.  
   - Metrics: Accuracy, Precision, Recall, and F1-score.  

---

## 📊 Final Model Performance
          precision    recall  f1-score   support

       0       0.97      0.99      0.98        71
       1       0.98      0.95      0.96        43

accuracy                           0.97       114

macro avg 0.97 0.97 0.97 114
weighted avg 0.97 0.97 0.97 114


✅ **Accuracy:** `97.36%`  
✅ Very few misclassifications (`[2 false negatives, 1 false positive]`).

---

## 🚀 How to Run
1. Clone this repository or download the notebook.  
2. Install required libraries:  
   ```bash
   pip install numpy pandas scikit-learn matplotlib
3. Download dataset from Kaggle and place it in your working directory.
4.Run the Python script / Jupyter notebook.

## Results & Insights

Linear SVM worked well but slightly limited on complex boundaries.

RBF Kernel SVM achieved better accuracy and generalization.

Tuning C and gamma significantly improved performance.

## Conclusion

SVM is a powerful classifier for both linear and non-linear data.

With proper feature scaling and hyperparameter tuning, SVM achieves high accuracy on medical datasets like Breast Cancer.

Final model achieved 97.36% accuracy, making it suitable for real-world applications.
