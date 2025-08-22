import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.DataFrame({
    "Age": [22, 25, 30, 35, 40, 45, 50],
    "Salary": [25000, 30000, 40000, 50000, 60000, 75000, 90000],
    "Experience": [1, 3, 5, 7, 10, 12, 15]
})

print("\n=== First 5 Rows of Data ===")
print(df.head())

print("\n=== Summary Statistics ===")
print(df.describe())  

print("\n=== Median Values ===")
print(df.median())

print("\n=== Variance ===")
print(df.var())

df.hist(figsize=(8, 5), bins=10, color="skyblue")
plt.suptitle("Histograms of Numerical Features")
plt.savefig("histograms.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Age", y="Salary", marker="o", color="blue")
plt.title("Age vs Salary")
plt.savefig("scatter_age_salary.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, y="Salary", color="orange")
plt.title("Box Plot of Salary")
plt.savefig("boxplot_salary.png", dpi=300, bbox_inches="tight")
plt.close()

corr_matrix = df.corr()
print("\n=== Correlation Matrix ===")
print(corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap_correlation.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n=== EDA Insights ===")
if corr_matrix.loc["Age", "Salary"] > 0.5:
    print("🔹 Strong positive correlation between Age and Salary.")
if df['Salary'].max() > 2 * df['Salary'].median():
    print("🔹 There might be potential outliers in Salary.")

print("🔹 Salary increases with Age and Experience in this dataset.")
