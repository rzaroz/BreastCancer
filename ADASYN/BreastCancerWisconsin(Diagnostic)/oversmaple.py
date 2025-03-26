import os
import time
import psutil
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder

start_time = time.time()

df_src = "../MainDatasets/BreastCancerWisconsin(Diagnostic).csv"
df = pd.read_csv(df_src)
class_counts = df['diagnosis'].value_counts()

class_counts.plot(kind='bar')
plt.title('Class distribution before oversample - ADASYN')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
plt.clf()

X = df.drop(["id", "Unnamed: 32", "diagnosis"], axis=1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["diagnosis"])

sm = ADASYN(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

final_y = []
for i_ in y_res:
    final_y.append(label_encoder.classes_[i_])

X_res["diagnosis"] = np.array(final_y)

X_res.to_csv("ADASYN-report/ADASYN-BreastCancerWisconsin(Diagnostic).csv", index=False)

class_counts = X_res["diagnosis"].value_counts()

class_counts.plot(kind='bar')
plt.title('Class distribution after oversample - ADASYN')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
plt.clf()


print("ADASYN - Usages")
end_time = time.time()
execution_time = end_time - start_time
print(f"Time: {execution_time} second")

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # تبدیل به مگابایت
print(f"Ram usage: {memory_usage} mb")

cpu_usage = process.cpu_percent(interval=1.0)
print(f"CPU usage: {cpu_usage}%")