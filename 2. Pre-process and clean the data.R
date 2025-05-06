library(mice)
###MICE包mice###MICE包对数据进行填补
data <- readxl::read_excel('最终数据表.xlsx')
colnames(data[29:85])
data[, 29:85] <- lapply(data[, 29:85], function(x) {
  x[x == 'N/A'] <- NA
  return(x)
})
data[, 29:85] <- lapply(data[, 29:85], as.numeric)
summary(data[29:85])
library(VIM)
aggr_plot<-aggr(data[29:85],col=c('navyblue','red'),numbers=TRUE,sortVars=TRUE,labels=names(data),
                cex.axis=.7,gap=3,ylab=c("Histogram of missing data","Pattern"))

tempData<-mice(data[29:85],m=5,maxit=50,meth='pmm',seed=500)
summary(tempData)
completedData<-complete(tempData,1)
data[29:85] <- completedData
data$TG 
data$glucose_min_avg
data$TyG <- log(data$TG*data$glucose_min_avg/2)


AP_ALL <- data
summary(AP_ALL)
# 转换为数值型
AP_ALL$first_day_height <- as.numeric(AP_ALL$first_day_height)
AP_ALL$first_day_weight <- as.numeric(AP_ALL$first_day_weight)

# 处理缺失值
AP_ALL$first_day_height[is.na(AP_ALL$first_day_height)] <- mean(AP_ALL$first_day_height, na.rm = TRUE)
AP_ALL$first_day_weight[is.na(AP_ALL$first_day_weight)] <- mean(AP_ALL$first_day_weight, na.rm = TRUE)
# 按 subject_id 分组，对其他列取最大值
AP_ALL <- AP_ALL %>%
  group_by(subject_id) %>%
  summarise_all(max, na.rm = TRUE)

# 定义需要转换为数值类型的实验室指标列
lab_columns <- c("hematocrit_max", "hemoglobin_max", "bun_max", "platelets_max", "wbc_max", 
                 "creatinine_max", "albumin_max", "glucose_min")

# 将实验室指标列转换为数值类型
AP_ALL[lab_columns] <- lapply(AP_ALL[lab_columns], as.numeric)

# 按 subject_id 分组，对其他列取最大值，去除重复的 subject_id
AP_ALL <- AP_ALL %>%
  group_by(subject_id) %>%
  summarise_all(max, na.rm = TRUE)

numeric_columns <- c("so2", "po2", "pco2", "fio2_chartevents", "fio2", "aado2", "aado2_calc", 
                     "pao2fio2ratio", "ph", "baseexcess", "bicarbonate", "totalco2", "hematocrit", 
                     "hemoglobin", "carboxyhemoglobin", "methemoglobin", "chloride", "calcium", 
                     "temperature", "potassium", "sodium", "lactate", "glucose")
AP_ALL[numeric_columns] <- lapply(AP_ALL[numeric_columns], as.numeric)

# Convert the charttime column to POSIXct type
AP_ALL$charttime <- ymd_hms(AP_ALL$charttime)

# Group by subject_id and calculate the maximum or minimum values
result <- AP_ALL %>%
  group_by(subject_id) %>%
  summarise(across(all_of(setdiff(numeric_columns, "fio2")), ~ max(.x, na.rm = TRUE), .names = "{.col}"),
            fio2 = min(fio2, na.rm = TRUE)) %>%
  ungroup()
# 计算 po2/fio2 * 100 的值
result$po2_fio2_ratio <- (result$po2 / result$fio2) * 100

# 根据计算结果添加 ARDS 列
result$ARDS <- ifelse(result$po2_fio2_ratio > 315, 1, 0)

# 查看结果
print(result)
table(result$ARDS)

result$ARDS[is.na(result$ARDS)] <- 0


# 将日期时间列转换为 POSIXct 类型
AP_ALL$charttime <- ymd_hms(AP_ALL$charttime)
summary(AP_ALL)
# 假设你的数据存储在一个名为 AP_ALL 的数据框中
# 将 creat, creat_low_past_48hr 和 creat_low_past_7day 转换为数值类型
AP_ALL$creat <- as.numeric(as.character(AP_ALL$creat))
AP_ALL$creat_low_past_48hr <- as.numeric(as.character(AP_ALL$creat_low_past_48hr))
AP_ALL$creat_low_past_7day <- as.numeric(as.character(AP_ALL$creat_low_past_7day))

# 创建新列 AKI
AP_ALL$AKI <- ifelse(
  (AP_ALL$creat - AP_ALL$creat_low_past_48hr >= 0.3) | (AP_ALL$creat_low_past_7day >= 1.5 * AP_ALL$creat),
  1,
  0
)

# 定义时间和类别型列
time_cols <- c("admittime", "dod", "admittime.2", "dischtime", "icu_intime", "icu_outtime", "charttime")
categorical_cols <- c("gender", "race", "first_hosp_stay", "first_icu_stay")

# 对数值列取最大值，对时间和类别型列取每组第一条记录
AP_ALL <- AP_ALL %>%
  group_by(subject_id) %>%
  summarise(
    across(all_of(time_cols), first),
    across(all_of(categorical_cols), first),
    across(where(is.numeric) & !any_of(c("subject_id", time_cols)), max, na.rm = TRUE)
  )
table(AP_ALL$AKI)
AP_d1 <- read.table('20250324AP患者第一天生命体征.csv', header = TRUE, sep = ",")
summary(AP_d1)
ids_to_remove <- AP_d1$subject_id[AP_d1$COF == 1]
tmp2 <- tmp1[!tmp1$subject_id %in% ids_to_remove,]
write.csv(tmp2, file = "20250323AP患者发生OF但第一天没有OF.csv", row.names = FALSE)
