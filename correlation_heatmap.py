from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# Data import
salary_data = r"C:\Users\patri\Downloads\salaryData.csv"
salary_df = pd.read_csv(salary_data)

# Feature engineering.
correlation = salary_df[['Age', 'Years of Experience', 'Salary']].corr()

# Plot heatmap
plt.figure(figsize = (8, 6))
sns.heatmap(correlation, annot = True, cmap = 'coolwarm', fmt = ".2f", square = True)
plt.title('Correlation Matrix: Age, Experience, and Salary')
plt.tight_layout()
plt.show()