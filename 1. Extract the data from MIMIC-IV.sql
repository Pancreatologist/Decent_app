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
   icd.*, -- 添加icustay_detail表的字段
   j.*
      -- 第1天的实验室结果
FROM mimiciv_hosp.admissions  AS ad
-- 修改连接条件，仅通过 subject_id 连接
JOIN mimiciv_derived.icustay_detail icd ON ad.subject_id = icd.subject_id
INNER JOIN mimiciv_hosp.patients  AS pa ON ad.subject_id = pa.subject_id
INNER JOIN acute_pancreatitis_patients app ON ad.subject_id = app.subject_id AND ad.hadm_id = app.hadm_id 

-- Join comorbidity data
LEFT JOIN mimiciv_derived.first_day_vitalsign j 
    ON icd.subject_id = j.subject_id
	  and (EXTRACT(EPOCH FROM j.charttime)-EXTRACT(EPOCH FROM ad.admittime) )<86400
    -- 修改条件：筛选出 h 中 charttime 在 ad 的 admittime 后 24 小时之后的信息
   
	and (EXTRACT(EPOCH FROM j.charttime)-EXTRACT(EPOCH FROM ad.admittime) )>86400
	and (EXTRACT(EPOCH FROM j.charttime)-EXTRACT(EPOCH FROM ad.admittime) )<86400*28

-- ... existing code ...
