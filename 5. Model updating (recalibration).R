library(tidymodels)
library(ggsci)
library(probably)
library(ResourceSelection)
library(dcurves)
library(gtsummary)
library(cowplot)
library(readxl)
library(writexl)

# 注意：以下代码引用了在前面文件中定义的变量和模型
# 请在运行此文件之前确保已经运行了前面的文件，或者定义了以下变量：
# 1. final_logistic_fit - 训练好的逻辑回归模型
# 2. final_svm_fit - 训练好的SVM模型
# 3. final_xgb_fit - 训练好的XGBoost模型
# 4. final_rf_fit - 训练好的随机森林模型
# 5. nnet_fit - 训练好的神经网络模型
# 6. final_lgbm_fit - 训练好的LightGBM模型
# 7. test_transformed - 转换后的测试集数据

# 以下代码需要在上述变量定义后才能运行
# 这里暂时注释掉

###logistic
#predictions_logistic <- predict(final_logistic_fit, new_data = test_transformed, type = "prob")
#predictions_logistic_preds <- test_transformed |> 
#  bind_cols(predictions_logistic )

#recal_object_lr <- test_transformed |> 
#  bind_cols(predictions_logistic) |> 
#  cal_estimate_logistic(truth = diagnosis,
#                        estimate = starts_with(".pred"),
#                        smooth = FALSE)

#recal_logistic_preds <- predict(final_logistic_fit, new_data = test_transformed, type = "prob")|> 
#  cal_apply(recal_object_lr) 

###svm
#predictions_svm <- predict(final_svm_fit, new_data = test_transformed, type = "prob")
#predictions_svm_preds <- test_transformed |> 
#  bind_cols(predictions_svm )
#recal_object_svm <- test_transformed |> 
#  bind_cols(predictions_svm) |> 
#  cal_estimate_logistic(truth = diagnosis,
#                        estimate = starts_with(".pred"),
#                        smooth = FALSE)
#recal_svm_preds <- predict(final_svm_fit, new_data = test_transformed, type = "prob")|> 
#  cal_apply(recal_object_svm) 

###xgboost
#predictions_xgb <- predict(final_xgb_fit, new_data = test_transformed, type = "prob")
#predictions_xgb_preds <- test_transformed |> 
#  bind_cols(predictions_xgb )
#recal_object_xgb <- test_transformed |> 
#  bind_cols(predictions_xgb) |> 
#  cal_estimate_logistic(truth = diagnosis,
#                        estimate = starts_with(".pred"),
#                        smooth = FALSE)
#recal_xgb_preds <- predict(final_xgb_fit, new_data = test_transformed, type = "prob")|> 
#  cal_apply(recal_object_xgb) 

###rf
#predictions_rf <- predict(final_rf_fit, new_data = test_transformed, type = "prob")
#predictions_rf_preds <- test_transformed |> 
#  bind_cols(predictions_rf )
#recal_object_rf <- test_transformed |> 
#  bind_cols(predictions_rf) |> 
#  cal_estimate_logistic(truth = diagnosis,
#                        estimate = starts_with(".pred"),
#                        smooth = FALSE)
#recal_rf_preds <- predict(final_rf_fit, new_data = test_transformed, type = "prob")|> 
#  cal_apply(recal_object_rf) 

###dnn
#predictions_dnn <- predict(nnet_fit, new_data = test_transformed, type = "prob")
#predictions_dnn_preds <- test_transformed |> 
#  bind_cols(predictions_dnn )
#recal_object_dnn <- test_transformed |> 
#  bind_cols(predictions_dnn) |> 
#  cal_estimate_logistic(truth = diagnosis,
#                        estimate = starts_with(".pred"),
#                        smooth = FALSE)
#recal_dnn_preds <- predict(nnet_fit, new_data = test_transformed, type = "prob")|> 
#  cal_apply(recal_object_dnn) 

###lighGBM
#predictions_lgbm <- predict(final_lgbm_fit, new_data = test_transformed, type = "prob")
#predictions_lgbm_preds <- test_transformed |> 
#  bind_cols(predictions_lgbm)
#recal_object_lgbm <- test_transformed |> 
#  bind_cols(predictions_lgbm) |> 
#  cal_estimate_logistic(truth = diagnosis,
#                        estimate = starts_with(".pred"),
#                        smooth = FALSE)
#recal_lgbm_preds <- predict(final_lgbm_fit, new_data = test_transformed, type = "prob")|> 
#  cal_apply(recal_object_lgbm) 


#recal_logistic_pred <- recal_logistic_preds  %>% select(.pred_1) %>% rename(recal_logistic = .pred_1)
#recal_svm_pred <- recal_svm_preds  %>% select(.pred_1) %>% rename(recal_svm = .pred_1)
#recal_xgb_pred <- recal_xgb_preds %>% select(.pred_1) %>% rename(recal_xgb = .pred_1)
#recal_rf_pred <- recal_rf_preds %>% select(.pred_1) %>% rename(recal_rf = .pred_1)
#recal_dnn_pred <- recal_dnn_preds  %>% select(.pred_1) %>% rename(recal_dnn = .pred_1)
#recal_lgbm_pred <- recal_lgbm_preds %>% select(.pred_1) %>% rename(recal_lgbm = .pred_1)
#recal_final_preds <- bind_cols(recal_logistic_pred, recal_svm_pred,
#                               recal_xgb_pred, recal_rf_pred,
#                               recal_dnn_pred, recal_lgbm_pred) %>% 
#  mutate(diagnosis=test_transformed$diagnosis)

#writexl::write_xlsx(recal_final_preds,'20250717_recal.xlsx')

results <- readxl::read_xlsx('20250717_recal带BISAP.xlsx')

results$diagnosis <- factor(results$diagnosis,levels = c(0, 1))

#-----3.校准曲线绘制------
library(scales)   # 提供刻度格式化工具

# 预测结果及真实标签汇总为一个数据框
calibration_data <- data.frame(
  Model = c(rep("Logistic Regression", length(results$recal_logistic)),
            rep("Random Forest", length(results$recal_rf)),
            rep("SVM", length(results$recal_svm)),
            rep("XGBoost", length(results$recal_xgb)),
            rep("LightGBM", length(results$recal_lgbm)),
            rep("Deep Neural Networks", length(results$recal_dnn)),
            rep("BISAP", length(results$BISAP))
  ),
  Probability = c(results$recal_logistic, 
                  results$recal_rf,
                  results$recal_svm,
                  results$recal_xgb,
                  results$recal_lgbm,
                  results$recal_dnn,
                  results$BISAP),
  Actual = as.numeric(c(results$diagnosis)) - 1  # 将因子转为数值
)

results$recalibrated_lr_probs <- predict(glm(results$diagnosis ~ results$recal_logistic, family = binomial),type = "response")
results$recalibrated_rf_probs <- predict(glm(results$diagnosis ~ results$recal_rf, family = binomial),type = "response")
results$recalibrated_svm_probs <- predict(glm(results$diagnosis ~ results$recal_svm, family = binomial),type = "response")
results$recalibrated_recal_xgb_probs <- predict(glm(results$diagnosis ~ results$recal_xgb, family = binomial),type = "response")
results$recalibrated_lgb_probs <- predict(glm(results$diagnosis ~ results$recal_lgbm, family = binomial),type = "response")
results$recalibrated_recal_dnn_probs <- predict(glm(results$diagnosis ~ results$recal_dnn, family = binomial),type = "response")
results$recalibrated_BISAP_probs <- predict(glm(results$diagnosis ~ results$BISAP, family = binomial),type = "response")
recalibration_data <- data.frame(
  Model = c(rep("Logistic Regression", length(results$recalibrated_lr_probs)),
            rep("Random Forest", length(results$recalibrated_rf_probs)),
            rep("SVM", length(results$recalibrated_svm_probs)),
            rep("XGBoost", length(results$recalibrated_recal_xgb_probs)),
            rep("LightGBM", length(results$recalibrated_lgb_probs)),
            rep("Deep Neural Networks", length(results$recalibrated_recal_dnn_probs)),
            rep("BISAP", length(results$recalibrated_BISAP_probs))
  ),
  Probability = c(results$recalibrated_lr_probs, 
                  results$recalibrated_rf_probs,
                  results$recalibrated_svm_probs,
                  results$recalibrated_recal_xgb_probs,
                  results$recalibrated_lgb_probs,
                  results$recalibrated_recal_dnn_probs,
                  results$recalibrated_BISAP_probs),
  Actual = as.numeric(c(results$diagnosis)) - 1  # 将因子转为数值
)

cal_plot <- cal_plot_logistic(recalibration_data, truth = Actual, estimate = Probability,
                              .by=Model, 
                              smooth =F
)  +scale_color_bmj()+ facet_wrap(~ Model, ncol =4)

cal_plot 
table(recalibration_data$Model)

recalibration_data$Model <- factor(recalibration_data$Model,
                                   levels = c("Logistic Regression","Random Forest","SVM","XGBoost",
                                              "LightGBM",
                                              "Deep Neural Networks","BISAP"))


calibration_data



# 定义计算各指标的函数
calculate_brier_score <- function(probabilities, actual) {
  mean((probabilities - actual)^2)
}
calculate_log_loss <- function(probabilities, actual) {
  # 将预测概率裁剪到 (0, 1) 范围内
  probabilities <- pmax(pmin(probabilities, 1 - 1e-10), 1e-10)
  -mean(actual * log(probabilities) + (1 - actual) * log(1 - probabilities))
}
calculate_calibration_slope <- function(probabilities, actual) {
  # 将预测概率裁剪到 (0, 1) 范围内
  probabilities <- pmax(pmin(probabilities, 1 - 1e-10), 1e-10)
  fit <- glm(actual ~ log(probabilities / (1 - probabilities)), family = binomial)
  coef(fit)[2]
}
# calculate_calibration_intercept 函数
calculate_calibration_intercept <- function(probabilities, actual) {
  # 将预测概率裁剪到 (0, 1) 范围内
  probabilities <- pmax(pmin(probabilities, 1 - 1e-10), 1e-10)
  # 使用 suppressWarnings 忽略 glm.fit 的警告
  suppressWarnings({
    fit <- glm(actual ~ 1, offset = log(probabilities / (1 - probabilities)), family = binomial)
    coef(fit)[1]
  })
}

calculate_hl_test <- function(probabilities, actual, n_groups = 10) {
  hoslem.test(actual, probabilities, g = n_groups)$p.value
}
calculate_ece <- function(probabilities, actual, n_bins = 10) {
  bin_indices <- cut(probabilities, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  ece <- 0
  for (i in levels(bin_indices)) {
    bin_samples <- bin_indices == i
    if (sum(bin_samples) > 0) {  # 确保分箱中有样本
      bin_prob <- mean(probabilities[bin_samples])
      bin_actual <- mean(actual[bin_samples])
      ece <- ece + abs(bin_prob - bin_actual) * sum(bin_samples) / length(actual)
    }
  }
  ece
}
#  calculate_mce 函数
calculate_mce <- function(probabilities, actual, n_bins = 10) {
  bin_indices <- cut(probabilities, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  mce <- 0
  for (i in levels(bin_indices)) {
    bin_samples <- bin_indices == i
    if (sum(bin_samples) > 0) {  # 确保分箱中有样本
      bin_prob <- mean(probabilities[bin_samples])
      bin_actual <- mean(actual[bin_samples])
      mce <- max(mce, abs(bin_prob - bin_actual))
    }
  }
  mce
}

# 计算每个模型的指标
metrics_summary <- calibration_data %>%
  group_by(Model) %>%
  summarise(
    Brier_Score = calculate_brier_score(Probability, Actual),
    Log_Loss = calculate_log_loss(Probability, Actual),
    Calibration_Slope = calculate_calibration_slope(Probability, Actual),
    Calibration_Intercept = calculate_calibration_intercept(Probability, Actual),
    #HL_Test_P_Value = calculate_hl_test(Probability, Actual),
    ECE = calculate_ece(Probability, Actual),
    MCE = calculate_mce(Probability, Actual)
  )

# 打印指标汇总表
print(metrics_summary)


# 保存结果到CSV文件
write.csv(metrics_summary, file = "calibration_metrics_recal_summary.csv", row.names = FALSE)


#-----4.DCA曲线绘制------
library(rmda)

dca_data <- data.frame(Result = as.factor(results$diagnosis), 
                       results$recal_logistic, 
                       results$recal_rf,
                       results$recal_svm,
                       results$recal_xgb,
                       results$recal_lgbm,
                       results$recal_dnn,
                       results$BISAP)

p <- dca(Result ~ results.recal_xgb + results.BISAP, 
         data = dca_data, 
         thresholds = seq(0, 0.2, by = 0.01), 
         label = list(results.recal_xgb = "Recalibrated XGboost Model",
                      results.BISAP= 'BISAP score')) %>%
  plot(smooth = TRUE)+scale_colour_jama()

p2 <- dca(Result ~ results.recal_xgb + results.BISAP+results.recal_logistic+
            results.recal_rf+results.recal_svm+results.recal_lgbm+results.recal_dnn, 
          data = dca_data, 
          thresholds = seq(0, 0.2, by = 0.01), 
          label = list(results.recal_xgb = "Recalibrated XGboost Model",
                       results.BISAP= 'BISAP score',
                       results.recal_logistic= "Recalibrated Logistic Regression",
                       results.recal_rf="Recalibrated Random Forest",
                       results.recal_svm="Recalibrated SVM",
                       results.recal_lgbm="Recalibrated LightGBM",
                       results.recal_dnn="Recalibrated Deep Neural Networks")) %>%
  plot(smooth = TRUE)+theme_cowplot(12)

#### CIC曲线
dca_data$Result <- as.numeric(dca_data$Result)-1
dca.result_recal_xgb <- decision_curve(Result ~ results.recal_xgb, 
                                       data = dca_data, 
                                       family = "binomial",
                                       thresholds = seq(0, 0.2, by = .01),
                                       bootstraps = 10)

plot_clinical_impact(dca.result_recal_xgb, col = "#7D5CC6FF",
                     population.size = 216,    #样本量1000
                     cost.benefit.axis = T,     #显示损失-收益坐标轴
                     n.cost.benefits= 8,
                     xlim=c(0,0.2),
                     confidence.intervals= T)


