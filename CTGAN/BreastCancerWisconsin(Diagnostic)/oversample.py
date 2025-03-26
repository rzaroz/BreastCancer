import os
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder

start_time = time.time()

# Load dataset
df = pd.read_csv("../MainDatasets/BreastCancerWisconsin(Diagnostic).csv")
df = df.drop(['Unnamed: 32', 'id'], axis=1)

# Encode target column
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])

# بررسی تعداد نمونه‌های هر کلاس
class_counts = df['diagnosis'].value_counts()
max_class_count = class_counts.max()

print("\nBefore Balancing:\n", class_counts)

# نمایش توزیع داده‌ها قبل از بالانس
class_counts.plot(kind='bar')
plt.title('Class Distribution Before Balancing - CTGAN')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# ایجاد داده‌های مصنوعی برای کلاس‌های نامتعادل
ctmodel = CTGAN(epochs=400, batch_size=40, pac=10, verbose=True)
ctmodel.fit(df)

generated_data = []
for cls in class_counts.index:
    current_count = class_counts[cls]
    needed_samples = max_class_count - current_count
    if needed_samples > 0:
        print(f"Generating {needed_samples} samples for class {cls}...")
        synthetic_df = ctmodel.sample(needed_samples)
        synthetic_df["diagnosis"] = cls
        generated_data.append(synthetic_df)

if generated_data:
    balanced_df = pd.concat([df] + generated_data, ignore_index=True)
else:
    balanced_df = df.copy()

new_class_counts = balanced_df["diagnosis"].value_counts()
print("\nAfter Balancing:\n", new_class_counts)

balanced_df.to_csv("CTGAN-report/CTGAN-BreastCancerWisconsin(Diagnostic).csv", index=False)

new_class_counts.plot(kind='bar')
plt.title('Class Distribution After Balancing - CTGAN')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# بررسی میزان مصرف منابع
end_time = time.time()
execution_time = end_time - start_time
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024
cpu_usage = process.cpu_percent(interval=1.0)

print(f"\nExecution Time: {execution_time:.2f} sec")
print(f"RAM Usage: {memory_usage:.2f} MB")
print(f"CPU Usage: {cpu_usage:.2f}%")
