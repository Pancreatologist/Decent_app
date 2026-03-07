Decent App: Machine Learning for Early Prediction of Organ Failure in Acute Pancreatitis
This repository contains the code and data for the manuscript:
"Development and validation of an interpretable machine learning model for early prediction of organ failure in acute pancreatitis patients: A multicenter cohort study" (under review).

The project develops and externally validates an XGBoost-based model to predict new‑onset organ failure within 28 days of admission using six routinely available clinical features. The model is accompanied by SHAP explanations and deployed as an interactive web‑based Shiny application.

Repository Structure
File	Description
1. Extract the data from MIMIC-IV.sql	SQL queries to extract the discovery cohort from the MIMIC‑IV database.
2. Pre-process and clean the data.R	Data cleaning, missing value imputation and TyG index calculation.
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
The discovery dataset about MIMIC-IV data is reposited in the PhysioNet (https://doi.org/10.13026/kpb9-mt58) without sharing it ourselves due to the license for these data (https://physionet.org/content/mimiciv/view-license/3.1/). Part of MIMIC-IV was granted access to the MIMIC-IV databases where one of the authors (Di Wu, No. 55097674) passed the Collaborative Institutional Training Initiative program exam giving them access to the database. The validation dataset, from Xiangya and Third Xiangya hospital, is available from the corresponding authors on reasonable request to the Ethics Committee of Xiangya hospital and Third Xiangya hospital. Once the request received, the corresponding authors will deliver to the the Ethics Committee. 

Citation
If you use this code or the model in your research, please cite our manuscript (once published). For now, you can reference this repository:

Wu D, Cai W, Chen C, et al. Development and validation of an interpretable machine learning model for early prediction of organ failure in acute pancreatitis patients: An international multicenter cohort study. 2025. GitHub: https://github.com/Pancreatologist/Decent_app

License
This project is licensed under the MIT License – see the LICENSE file.

Contact
Di Wu (First author): wudikdfz@qq.com

