Decent App: Machine Learning for Early Prediction of Organ Failure in Acute Pancreatitis
This repository contains the code and data for the manuscript:
"Development and validation of an interpretable machine learning model for early prediction of organ failure in acute pancreatitis patients: A multicenter cohort study" (under review).

The project develops and externally validates an XGBoost-based model to predict new‑onset organ failure within 28 days of admission using six routinely available clinical features. The model is accompanied by SHAP explanations and deployed as an interactive web‑based Shiny application.

Repository Structure
File	Description
1. Extract the data from MIMIC-IV.sql	SQL queries to extract the discovery cohort from the MIMIC‑IV database.
2. Pre-process and clean the data.R	Data cleaning, missing value imputation, winsorization, and TyG index calculation.
3. Feature selection and machine learning	Feature selection (Boruta + LASSO) and training of six models (elastic net, SVM, random forest, XGBoost, LightGBM, neural network).
4. Validation and visualization.R	External validation, ROC curves, calibration plots, decision curve analysis, and SHAP interpretation.
5. Model updating (recalibration).R	Logistic recalibration of models on the validation cohort.
discovery cohort data.xlsx	Processed discovery cohort data (after imputation and cleaning).
LICENSE	MIT License.
README.md	This file.
Requirements
R version ≥ 4.3.3

Data extraction
Run 1. Extract the data from MIMIC-IV.sql on a local copy of MIMIC‑IV (requires PhysioNet access).

Preprocessing
Run 2. Pre-process and clean the data.R to generate the cleaned discovery cohort.

Feature selection & model training
Run 3. Feature selection and machine learning to identify key predictors and train all models.

Validation & visualisation
Run 4. Validation and visualization.R to evaluate model performance on the validation cohort and generate figures.

Model recalibration
Run 5. Model updating (recalibration).R to recalibrate models and obtain the final XGBoost model.

Web application
The interactive Shiny app is available at: https://wudikdfz.shinyapps.io/Decentapp
Source code for the app is in a separate repository: Pancreatologist/Decent_app.

Data Availability
MIMIC‑IV is publicly available but requires registration. The SQL script is provided for replication.

Processed discovery cohort data is included as discovery cohort data.xlsx for transparency.

The validation cohort (Xiangya hospitals) data cannot be made public due to institutional policies; anonymized data are available upon reasonable request to the corresponding authors.

Citation
If you use this code or the model in your research, please cite our manuscript (once published). For now, you can reference this repository:

Wu D, Cai W, Chen C, et al. Development and validation of an interpretable machine learning model for early prediction of organ failure in acute pancreatitis patients: An international multicenter cohort study. 2025. GitHub: https://github.com/Pancreatologist/Decent_app

License
This project is licensed under the MIT License – see the LICENSE file.

Contact
Di Wu (First author): wudikdfz@qq.com

