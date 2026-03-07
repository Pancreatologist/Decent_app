WITH diseases AS (
    -- Filter out disease information for acute pancreatitis
    SELECT icd_code, icd_version, long_title
    FROM mimiciv_hosp.d_icd_diagnoses
    WHERE long_title ILIKE '%Acute pancreatitis%'
),
acute_pancreatitis_patients AS (
    -- Filter out patients with acute pancreatitis
    SELECT d.subject_id, d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN diseases h ON d.icd_code = h.icd_code AND d.icd_version = h.icd_version
)
-- Extract basic information of patients with acute pancreatitis
SELECT
    ad.subject_id,
    ad.hadm_id,
    ad.admittime,
    icd.*,
    j.*
FROM mimiciv_hosp.admissions AS ad
INNER JOIN mimiciv_hosp.patients pa ON ad.subject_id = pa.subject_id
INNER JOIN acute_pancreatitis_patients app ON ad.subject_id = app.subject_id AND ad.hadm_id = app.hadm_id
INNER JOIN mimiciv_derived.icustay_detail icd ON ad.subject_id = icd.subject_id
LEFT JOIN mimiciv_derived.first_day_vitalsign j
    ON icd.subject_id = j.subject_id
    AND (EXTRACT(EPOCH FROM j.charttime) - EXTRACT(EPOCH FROM ad.admittime)) < 86400 * 28
    AND EXTRACT(EPOCH FROM j.charttime) - EXTRACT(EPOCH FROM ad.admittime) > 86400

-- ... existing code ...

WITH diseases AS (
    -- Filter out disease information for acute pancreatitis
    SELECT icd_code, icd_version, long_title 
    FROM mimiciv_hosp.d_icd_diagnoses 
    WHERE long_title ILIKE '%Acute pancreatitis%'
),
acute_pancreatitis_patients AS (
    -- Filter out patients with acute pancreatitis
    SELECT d.subject_id, d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN diseases h ON d.icd_code = h.icd_code AND d.icd_version = h.icd_version
)
-- Extract basic information of patients with acute pancreatitis
SELECT   
   ad.subject_id,
   ad.hadm_id,
   ad.admittime,
   -- First-day GCS score in ICU
   gcs.gcs AS first_day_gcs,
   -- Comorbidity data
   charlson.charlson_comorbidity_index,
      -- 第1天的实验室结果
	  h.hematocrit_max,
        h.hemoglobin_max,
        h.bun_max,
        h.platelets_max,
        h.wbc_max,
        h.creatinine_max,
        h.albumin_max,
        h.glucose_min
FROM mimiciv_hosp.admissions  AS ad
INNER JOIN mimiciv_hosp.patients  AS pa ON ad.subject_id = pa.subject_id
INNER JOIN acute_pancreatitis_patients app ON ad.subject_id = app.subject_id AND ad.hadm_id = app.hadm_id
-- Join first-day GCS score information in ICU
LEFT JOIN (
    WITH t1 AS (
        SELECT i.subject_id, i.stay_id, g.charttime, g.gcs,
               ROW_NUMBER () OVER(PARTITION BY g.SUBJECT_ID ORDER BY g.charttime) AS CHARTTIME_RANK
        FROM mimiciv_derived.icustay_detail i
        INNER JOIN mimiciv_derived.gcs g ON i.stay_id = g.stay_id
        WHERE g.charttime BETWEEN i.icu_intime AND mimiciv_derived.DATETIME_ADD(i.icu_intime, INTERVAL '24' HOUR)
        AND g.gcs is not NULL
    )
    SELECT * FROM t1 WHERE CHARTTIME_RANK = 1
) gcs ON ad.subject_id = gcs.subject_id 
-- Join comorbidity data
LEFT JOIN mimiciv_derived.charlson charlson ON ad.subject_id = charlson.subject_id AND ad.hadm_id = charlson.hadm_id

-- Join comorbidity data
LEFT JOIN mimiciv_derived.first_day_lab h ON ad.subject_id = h.subject_id 


-- ... existing code ...


WITH diseases AS (
    -- Filter out disease information for acute pancreatitis
    SELECT icd_code, icd_version, long_title 
    FROM mimiciv_hosp.d_icd_diagnoses 
    WHERE long_title ILIKE '%Acute pancreatitis%'
),
acute_pancreatitis_patients AS (
    -- Filter out patients with acute pancreatitis
    SELECT d.subject_id, d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN diseases h ON d.icd_code = h.icd_code AND d.icd_version = h.icd_version
),
t1 as (
    -- 子查询用于筛选 itemid 为 51000 且 valuenum 不为空的记录
    WITH filtered_labevents AS (
        SELECT subject_id, hadm_id, charttime, valuenum
        FROM mimiciv_hosp.labevents
        WHERE itemid IN (51000) AND valuenum IS NOT NULL
    )
    -- 筛选出每个 subject_id 对应的 valuenum 最小的记录
    SELECT subject_id, hadm_id, charttime, valuenum
    FROM (
        SELECT 
            subject_id, 
            hadm_id, 
            charttime, 
            valuenum,
            -- 为每个 subject_id 的记录按 valuenum 升序排序并编号
            ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY valuenum) as rn
        FROM filtered_labevents
    ) ranked
    WHERE rn = 1 -- 只保留 valuenum 最小的记录
)
SELECT   
    ad.subject_id,
    ad.hadm_id,
    ad.admittime,
    -- 第1天的实验室结果
    t1.*
FROM mimiciv_hosp.admissions  AS ad
INNER JOIN mimiciv_hosp.patients  AS pa ON ad.subject_id = pa.subject_id
INNER JOIN acute_pancreatitis_patients app ON ad.subject_id = app.subject_id AND ad.hadm_id = app.hadm_id
-- 提取数据
LEFT JOIN mimiciv_icu.icustays h ON ad.subject_id = h.subject_id
LEFT JOIN t1 ON ad.subject_id = t1.subject_id AND ad.hadm_id = t1.hadm_id;
