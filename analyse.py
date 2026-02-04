# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file
df = pd.read_csv("data.csv")

# Display dataset
print("Dataset Preview:")
print(df.head())

# Basic Data Analysis
average_marks = df["Marks"].mean()
print("\nAverage Marks:", average_marks)

print("\nStatistical Summary:")
print(df.describe())

# -------------------- Data Visualization --------------------

# 1. Bar Chart - Marks of Students
plt.figure()
plt.bar(df["Name"], df["Marks"])
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Marks of Students")
plt.show()

# 2. Scatter Plot - Attendance vs Marks
plt.figure()
plt.scatter(df["Attendance"], df["Marks"])
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.title("Attendance vs Marks")
plt.show()

# 3. Heatmap - Correlation Matrix
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

