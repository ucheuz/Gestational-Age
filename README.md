# Gestational-Age
A machine learning algorithm to predict the gestational age (GA) of a fetus based on microstructural measures and volumes of fetal brain layers.The project includes:
  A Python Script (main.py) that runs the model.
  A Juypter Notebook (Gestational Age.ipynb) Task 1 with exploratory data analyis, model training and conclusions about the model at each stage.

# Description
In this project, I developed a machine learning model to predict the gestational age (GA) of a fetus based on microstructural measures and volumes of fetal brain layers. The model utilized features from these measurements to make predictions. To implement the model, I manually built a Ridge regression algorithm using NumPy, ensuring that only the feature weights (and not the intercept) were penalized during the optimization. The model was evaluated using root mean squared error (RMSE) to assess its accuracy on both training and test datasets. The performance was further compared to a standard Ridge regression implementation from scikit-learn. The final implementation involved plotting predicted vs. true GA values, and suggested improvements were based on the RMSE values and the resulting fit.

# How to run
Option 1: Running the Python Script: Can be done directly in terminal. This will train the model and output predictions and accuracy scores.

Option 2: Running the Juypter Notebook: Download and open Gestational Age.ipynb and run the cells in Task 1 step by step.

# Dataset
File: datasets/dataset-fetal-brain-reg.csv
Description: The dataset contains contains microstructural measures and volumes of fetal brain layers, followed by age at scan (GA) in the last column.

# Results
Training RMSE:  0.64
Test RMSE:  0.77
Conclusion: As the root mean squared error of the test set is greater than that of the training set we can say the model performs better on the training set than on the test set, suggesting it may be slightly overfitting.

# Contributors
Kosiasochukwu Uchemudi Uzoka - Author
