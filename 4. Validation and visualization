library(tidymodels)
library(ggsci)
library(probably)
library(ResourceSelection)
library(dcurves)
library(gtsummary)
library(cowplot)
results <- readxl::read_xlsx('20250711带BISAP.xlsx')

results$diagnosis <- factor(results$diagnosis,levels = c(0, 1))

#-----1.混淆矩阵------
###Logistic 预测结果#######
pred_lr <- ifelse(results$logistic < 0.5, 1, 0)   

#  Logistic 创建混淆矩阵  

confusion_matrix_lr <- caret::confusionMatrix(
  factor(pred_lr,levels = c(0, 1),labels = c("0","1")), 
  results$diagnosis,
  positive = "1")   
print(confusion_matrix_lr) 


# 获取指标
metrics_lr <- data.frame(
  Accuracy = confusion_matrix_lr$overall["Accuracy"],
  Precision = confusion_matrix_lr$byClass["Precision"],
  Recall = confusion_matrix_lr$byClass["Recall"],
  F1 = confusion_matrix_lr$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$logistic)
)
metrics_lr
# 转置为垂直格式
final_metrics_lr <- metrics_lr %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_lr, "lr_metrics.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_lr <- as.data.frame(confusion_matrix_lr$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_lr <- ggplot(conf_matrix_df_lr, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for Logistic Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_lr

# 保存图片
ggsave("./混淆矩阵_lr.pdf", conf_plot_lr, width = 6, height = 6)

### 随机森林模型 预测结果#####
# 在测试集上评估模型的区分能力
prob_rf <- ifelse(results$rf > 0.5, 1, 0)  
#  随机森林模型 创建混淆矩阵  
confusion_matrix_rf <- caret::confusionMatrix(
  factor(prob_rf,levels = c(0, 1),labels = c("0","1")), 
  results$diagnosis,
  positive = "1")   
print(confusion_matrix_rf) 



# 获取指标
metrics_rf <- data.frame(
  Accuracy = confusion_matrix_rf$overall["Accuracy"],
  Precision = confusion_matrix_rf$byClass["Precision"],
  Recall = confusion_matrix_rf$byClass["Recall"],
  F1 = confusion_matrix_rf$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$rf)
)
metrics_rf
# 转置为垂直格式
final_metrics_rf <- metrics_rf %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_rf, "rf_metrics.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_rf <- as.data.frame(confusion_matrix_rf$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_rf <- ggplot(conf_matrix_df_rf, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for Random Forest Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_rf

# 保存图片
ggsave("./混淆矩阵_rf.pdf", conf_plot_lr, width = 6, height = 6)

###支持向量机 预测结果#######
prob_svm <- ifelse(results$svm > 0.5, 1, 0)  

#  支持向量机 创建混淆矩阵  
confusion_matrix_svm <- caret::confusionMatrix(
  factor(prob_svm,levels =c("0", "1"),labels = c("0","1")), 
  results$diagnosis,
  positive = "1")   # 训练集
print(confusion_matrix_svm) 


# 获取指标
metrics_svm <- data.frame(
  Accuracy = confusion_matrix_svm$overall["Accuracy"],
  Precision = confusion_matrix_svm$byClass["Precision"],
  Recall = confusion_matrix_svm$byClass["Recall"],
  F1 = confusion_matrix_svm$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = prob_svm)
)
metrics_svm
# 转置为垂直格式
final_metrics_svm <- metrics_svm %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_svm, "svm_metrics.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_svm <- as.data.frame(confusion_matrix_svm$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_svm <- ggplot(conf_matrix_df_svm, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for SVM Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_svm

# 保存图片
ggsave("./混淆矩阵_svm.pdf", conf_plot_lr, width = 6, height = 6)



###XGboost 预测结果#######
pred_xgb <- ifelse(results$xgb > 0.5, 1, 0)  
# 创建混淆矩阵  
confusion_matrix_xgb <- caret::confusionMatrix(factor(pred_xgb,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_xgb) 

# 获取指标
metrics_xgb <- data.frame(
  Accuracy = confusion_matrix_xgb$overall["Accuracy"],
  Precision = confusion_matrix_xgb$byClass["Precision"],
  Recall = confusion_matrix_xgb$byClass["Recall"],
  F1 = confusion_matrix_xgb$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$xgb)
)
metrics_xgb 
# 转置为垂直格式
final_metrics_xgb <- metrics_xgb %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_xgb, "xgb_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_xgb <- as.data.frame(confusion_matrix_xgb$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_xgb <- ggplot(conf_matrix_df_xgb, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for XGboost Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_xgb



# 保存图片
ggsave("./混淆矩阵_xgb.pdf", conf_plot_xgb, width = 6, height = 6)

###LightGBM 预测结果#######
pred_lgb <- ifelse(results$lgbm > 0.5, 1, 0)  
# 创建混淆矩阵  
confusion_matrix_lgb <- caret::confusionMatrix(factor(pred_lgb,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_lgb) 

# 获取指标
metrics_lgb <- data.frame(
  Accuracy = confusion_matrix_lgb$overall["Accuracy"],
  Precision = confusion_matrix_lgb$byClass["Precision"],
  Recall = confusion_matrix_lgb$byClass["Recall"],
  F1 = confusion_matrix_lgb$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor =  results$lgbm)
)
metrics_lgb 
# 转置为垂直格式
final_metrics_lgb <- metrics_lgb %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_lgb, "lgb_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_lgb <- as.data.frame(confusion_matrix_lgb$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_lgb <- ggplot(conf_matrix_df_lgb, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for LightGBM Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_lgb

ggsave("./混淆矩阵_Lgb.pdf", conf_plot_lgb, width = 6, height = 6)


###DNN 预测结果#######
pred_DNN <- ifelse(results$dnn > 0.5, 1, 0) 

# 创建混淆矩阵  
confusion_matrix_DNN <- caret::confusionMatrix(factor(pred_DNN,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_DNN) 

# 获取指标
metrics_DNN <- data.frame(
  Accuracy = confusion_matrix_DNN$overall["Accuracy"],
  Precision = confusion_matrix_DNN$byClass["Precision"],
  Recall = confusion_matrix_DNN$byClass["Recall"],
  F1 = confusion_matrix_DNN$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$dnn)
)
metrics_DNN 
# 转置为垂直格式
final_metrics_DNN <- metrics_DNN %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_DNN, "DNN_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_DNN <- as.data.frame(confusion_matrix_DNN$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_DNN <- ggplot(conf_matrix_df_DNN, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for Deep Neural Networks Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_DNN



# 保存图片
ggsave("./混淆矩阵_DNN.pdf", conf_plot_DNN, width = 6, height = 6)


#-----2.单次ROC曲线绘制------
# 计算ROC对象
library(PRROC)
library(ROCR)
library(pROC)
roc_lr <- roc(results$diagnosis, as.numeric(results$logistic))
roc_rf <- roc(results$diagnosis, as.numeric(results$rf))
roc_svm <- roc(results$diagnosis,as.numeric(results$svm))
roc_xgb <- roc(results$diagnosis, as.numeric(results$xgb))
roc_lgb <- roc(results$diagnosis, as.numeric(results$lgbm))
roc_dnn <- roc(results$diagnosis, as.numeric(results$dnn))
roc_bisap <- roc(results$diagnosis, as.numeric(results$BISAP))


# 计算AUC值
auc_lr <- roc_lr$auc 
auc_rf <- roc_rf$auc 
auc_svm <- roc_svm$auc 
auc_xgb <- roc_xgb$auc 
auc_lgb <- roc_lgb$auc 
auc_dnn <- roc_dnn$auc 
auc_bisap <- roc_bisap$auc
ci_lr <- ci.auc(auc_lr)
ci_rf <- ci.auc(auc_rf)
ci_svm <- ci.auc(auc_svm)
ci_xgb <- ci.auc(auc_xgb)
ci_lgb <- ci.auc(auc_lgb)
ci_dnn <- ci.auc(auc_dnn)
ci_bisap <- ci.auc(auc_bisap)


# 创建数据框，包含每个模型的 AUC 值和 ROC 对象
roc_data <- list(
  Model = c("Logistic Regression","Random Forest","SVM", 
            "XGBoost","LightGBM",
            'Deep Neural Networks',
            'BISAP'),
  AUC = c(auc_lr,auc_rf,auc_svm, auc_xgb,auc_lgb,
          auc_dnn, auc_bisap),  # 实际模型的 AUC 值
  CI_Lower = c(ci_lr[1], ci_rf[1], ci_svm[1], ci_xgb[1], ci_lgb[1], ci_dnn[1],ci_bisap[1]),
  CI_Upper = c(ci_lr[3], ci_rf[3], ci_svm[3], ci_xgb[3], ci_lgb[3], ci_dnn[3],ci_bisap[3]),
  ROC = list(roc_lr,roc_rf,roc_svm, roc_xgb,roc_lgb,
             roc_dnn,roc_bisap)  # 实际模型的 ROC 对象
)

roc_data$Model <- factor(roc_data$Model,
                         levels = c("Logistic Regression","XGBoost",
                                    "Random Forest","LightGBM",
                                    "SVM","Deep Neural Networks",'BISAP'))
# 提取 ROC 曲线的数据并转换为数据框
roc_curves <- lapply(seq_along(roc_data$ROC), function(i) {
  roc_obj <- roc_data$ROC[[i]]
  data.frame(
    Sensitivity = roc_obj$sensitivities,
    Specificity = roc_obj$specificities,
    Model = roc_data$Model[i]
  )
})

# 将列表合并为一个数据框
roc_curves <- do.call(rbind, roc_curves)

# 对 ROC 对象进行平滑处理
smoothed_rocs <- lapply(roc_data$ROC, smooth, method = "density")  # 使用密度平滑方法

# 添加 AUC 值到图例标签
roc_data$Label <- paste(roc_data$Model, " (AUC [95%CI]= ", round(roc_data$AUC, 3),", ",
                        round(roc_data$CI_Lower, 3),"-",
                        round(roc_data$CI_Upper, 3), ")", sep = "")

# 绘制平滑 ROC 曲线
roc_plot <- ggroc(smoothed_rocs, linetype = 1, size = 1.2) +
  scale_color_bmj(labels = roc_data$Label) +  # 使用 Lancet 颜色主题并添加 AUC 标签
  labs(title = "ROC Curves for Models",
       x = "1 - Specificity",
       y = "Sensitivity") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(ncol = 1, byrow = TRUE))  # 图例分两列显示

# 显示图形
roc_plot

# 保存图片
ggsave("./单次ROC.pdf", roc_plot, width = 5, height = 6)

#-----3.校准曲线绘制------
library(scales)   # 提供刻度格式化工具

# 预测结果及真实标签汇总为一个数据框
calibration_data <- data.frame(
  Model = c(rep("Logistic Regression", length(results$logistic)),
            rep("Random Forest", length(results$rf)),
            rep("SVM", length(results$svm)),
            rep("XGBoost", length(results$xgb)),
            rep("LightGBM", length(results$lgbm)),
            rep("Deep Neural Networks", length(results$dnn)),
            rep("BISAP", length(results$BISAP))
  ),
  Probability = c(results$logistic, 
                  results$rf,
                  results$svm,
                  results$xgb,
                  results$lgbm,
                  results$dnn,
                  results$BISAP),
  Actual = as.numeric(c(results$diagnosis)) - 1  # 将因子转为数值
)


results$recalibrated_lr_probs <- predict(glm(results$diagnosis ~ qlogis(results$logistic), family = binomial))
results$recalibrated_rf_probs <- predict(glm(results$diagnosis ~ qlogis(results$rf), family = binomial))
results$recalibrated_svm_probs <- predict(glm(results$diagnosis ~ qlogis(results$svm), family = binomial))
results$recalibrated_xgb_probs <- predict(glm(results$diagnosis ~ qlogis(results$xgb), family = binomial))
results$recalibrated_lgb_probs <- predict(glm(results$diagnosis ~ qlogis(results$lgbm), family = binomial))
results$recalibrated_DNN_probs <- predict(glm(results$diagnosis ~ qlogis(results$dnn), family = binomial))
results$recalibrated_BISAP_probs <- predict(glm(results$diagnosis ~ qlogis(results$BISAP), family = binomial))

recalibration_data <- data.frame(
  Model = c(rep("Logistic Regression", length(results$recalibrated_lr_probs)),
            rep("Random Forest", length(results$recalibrated_rf_probs)),
            rep("SVM", length(results$recalibrated_svm_probs)),
            rep("XGBoost", length(results$recalibrated_xgb_probs)),
            rep("LightGBM", length(results$recalibrated_lgb_probs)),
            rep("Deep Neural Networks", length(results$recalibrated_DNN_probs)),
            rep("BISAP", length(results$recalibrated_BISAP_probs))
  ),
  Probability = c(results$recalibrated_lr_probs, 
                  results$recalibrated_rf_probs,
                  results$recalibrated_svm_probs,
                  results$recalibrated_xgb_probs,
                  results$recalibrated_lgb_probs,
                  results$recalibrated_DNN_probs,
                  results$recalibrated_BISAP_probs),
  Actual = as.numeric(c(results$diagnosis)) - 1  # 将因子转为数值
)

cal_plot <- cal_plot_logistic(calibration_data, truth = Actual, estimate = Probability
                  ,.by=Model,smooth =F
)  +scale_color_bmj()+ facet_wrap(~ Model, ncol =4)

cal_plot 


calibration_data$Model <- factor(calibration_data$Model,
                                 levels = c("Logistic Regression","Random Forest","SVM","XGBoost",
                                            "LightGBM",
                                            "Deep Neural Networks","BISAP","SIRS"))


#校准曲线
calibration_plot <- ggplot(calibration_data, aes(x = Probability, y = Actual, 
                                                 color = Model)) + 
  geom_smooth(method = "loess", se = FALSE, linewidth = 1.5, span = 0.9) +  
  # 方法有 loess、gam 
  # 增加 span 参数,较大的 span 值会产生更平滑的曲线，而较小的 span 值会使曲线更贴近数据点 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", linewidth = 1) + 
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) + 
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) + 
  labs( 
    title = "Calibration Curves", 
    x = "Actual Probability", 
    y = "Observed Proportion" 
  ) + 
  scale_color_manual(values = c("#2A6EBBFF",  
                                "#F0AB00FF",  
                                "#C50084FF",  
                                "#7D5CC6FF",  
                                "#E37222FF", 
                                "#69BE28FF"  )) + 
  theme_minimal() + 
  theme( 
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"), 
    axis.title = element_text(size = 14), 
    axis.text = element_text(size = 12), 
    legend.position = "bottom", 
    legend.title = element_text(size = 12), 
    legend.text = element_text(size = 10), 
    axis.line = element_line(colour = "black") 
  )
calibration_plot

ggsave("./校准曲线.pdf", calibration_plot, width = 6, height = 6)


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
write.csv(metrics_summary, file = "calibration_metrics_summary.csv", row.names = FALSE)


#-----4.DCA曲线绘制------
library(rmda)

dca_data <- data.frame(Result = as.factor(results$diagnosis), 
                       results$logistic, 
                       results$rf,
                       results$svm,
                       results$xgb,
                       results$lgbm,
                       results$dnn,
                       results$BISAP)

p <- dca(Result ~ results.xgb + results.BISAP, 
    data = dca_data, 
    thresholds = seq(0, 0.2, by = 0.01), 
    label = list(results.xgb = "XGboost Model",
                 results.BISAP= 'BISAP score')) %>%
  plot(smooth = TRUE)+scale_colour_jama()

p2 <- dca(Result ~ results.xgb + results.BISAP+results.logistic+
            results.rf+results.svm+results.lgbm+results.dnn, 
         data = dca_data, 
         thresholds = seq(0, 0.2, by = 0.01), 
         label = list(results.xgb = "XGboost Model",
                      results.BISAP= 'BISAP score',
                      results.logistic= "Logistic Regression",
                      results.rf="Random Forest",
                      results.svm="SVM",
                      results.lgbm="LightGBM",
                      results.dnn="Deep Neural Networks")) %>%
  plot(smooth = TRUE)+theme_cowplot(12)

#### CIC曲线
dca_data$Result <- as.numeric(dca_data$Result)-1
dca.result_xgb <- decision_curve(Result ~ results.xgb, 
                                 data = dca_data, 
                                 family = "binomial",
                                 thresholds = seq(0, 0.2, by = .01),
                                 bootstraps = 10)

plot_clinical_impact(dca.result_xgb, col = "#7D5CC6FF",
                     population.size = 216,    #样本量1000
                     cost.benefit.axis = T,     #显示损失-收益坐标轴
                     n.cost.benefits= 8,
                     xlim=c(0,0.2),
                     confidence.intervals= T)



#-----5.shap可视化-----

#加载模型
library(kernelshap)
library(shapviz)
library(ggplot2)

#分组变量转为因子
data <- train_transformed
data$diagnosis <- as.factor(data$diagnosis)
data_test <- test_transformed
data_test$diagnosis <- as.factor(data_test$diagnosis)
str(data)
#分离特征和应变量
target <- "diagnosis"
features <- data[, -which(names(data) == "diagnosis")]
# 设置样本量和准备数据
n <- 1429  # 样本量
testdata_matrix <- as.matrix(data_test[1:n, -1])
bg_X_matrix <- as.matrix(data[1:n, -1])  # 背景数据集

#xgbmodel
explain_kernel_dnn <- kernelshap(final_xgb_fit, 
                                 testdata_matrix, 
                                 bg_X = bg_X_matrix)  #有验证集的时候用验证集
shap_value_dnn <- shapviz(explain_kernel_dnn,
                          data[1:n, -1], which_class = 2,
                          interactions=TRUE) 
sv_interaction(shap_value_dnn)
sv_importance(shap_value_dnn)


xvars <- c('platelets','wbc','albumin',"bun","ptt","TyG") 


sv_dependence(shap_value_dnn,
              v= 'TyG')

sv_force(shap_value_dnn,row_id=1)
p1 <- sv_waterfall(shap_value_dnn,row_id=1L)
p2 <- sv_importance(shap_value_dnn,kind="beeswarm")
p3 <- sv_dependence(shap_value_dnn,xvars)
(p1+p2)/p3

heatmap_shap <- shap_value_dnn$X 
corr <- round(cor(heatmap_shap), 1) 

# 创建一个对角线为NA的相关系数矩阵
corr_no_diag <- corr
diag(corr_no_diag) <- NA

# 创建一个对角线为1的p值矩阵
res <- cor.mtest(heatmap_shap, conf.level = .95)
p <- res$p
diag(p) <- 1

# 使用pie方法绘制，标签贴左边边框
corrplot(corr_no_diag, method = "pie", type = "upper", 
         p.mat = p, sig.level = c(.001, .01, .05), 
         insig = "label_sig", pch.cex = 1.2, 
         pch.col = 'black', diag = FALSE, 
         mar = c(0, 0, 0, 0), tl.cex = 0.8)

