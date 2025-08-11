# ElevateLabTask4
---

## 📌 Steps in the Project
1. Load the Breast Cancer dataset from Scikit-learn.
2. Split into training and testing sets.
3. Standardize features so all values are on the same scale.
4. Train the logistic regression model.
5. Make predictions and check model performance.
6. Plot the ROC curve to see how well it separates the two classes.

---

---
Interview Questions and  Answers
1. How does logistic regression differ from linear regression?
Linear regression predicts numbers. Logistic regression predicts probabilities for categories.

2. What is the sigmoid function?
A curve that squashes any number into a range between 0 and 1, used to represent probabilities.

3. What is precision vs recall?

Precision: Of the positive predictions, how many were actually positive.

Recall: Of all actual positives, how many we correctly found.

4. What is the ROC-AUC curve?
A graph showing the trade-off between finding positives (recall) and making false alarms (false positives).
AUC measures the area under this curve — higher means better.

5. What is the confusion matrix?
A simple table showing counts of:
True Positives (TP),True Negatives (TN),False Positives (FP),False Negatives (FN)

6. What happens if classes are imbalanced?
The model might always predict the majority class. Accuracy becomes misleading — focus on precision, recall, and AUC instead.

7. How do you choose the threshold?
By looking at the ROC curve or Precision-Recall curve and picking a value that balances missed detections vs false alarms based on the problem.

8. Can logistic regression be used for multi-class problems?
Yes — using methods like "One vs Rest" or multinomial logistic regression.
