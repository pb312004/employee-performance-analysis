import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set visualization style
sns.set(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.autolayout': True})

# Load dataset
df = pd.read_excel("employee_data.xlsx", sheet_name="Performance")

# --- Basic Statistics ---
print("Top Performers by Sales:")
print(df.sort_values(by='Sales', ascending=False).head(3))
print("\nAverage Task Completion:", df['Tasks_Completed'].mean())
print("Average Attendance:", df['Attendance (%)'].mean())
print("\nSummary Statistics:")
print(df.describe())

# --- Correlation Heatmap ---
numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Between Performance Metrics", fontsize=16)
plt.show()

# --- Distribution Plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Sales'], kde=True, ax=axes[0], color='dodgerblue', edgecolor='black')
axes[0].set_title('Sales Distribution', fontsize=14)
axes[0].set_xlabel('Sales Amount')
axes[0].set_ylabel('Count')
sns.histplot(df['Tasks_Completed'], kde=True, ax=axes[1], color='mediumseagreen', edgecolor='black')
axes[1].set_title('Tasks Completed Distribution', fontsize=14)
axes[1].set_xlabel('Tasks Completed')
axes[1].set_ylabel('Count')
sns.histplot(df['Attendance (%)'], kde=True, ax=axes[2], color='tomato', edgecolor='black')
axes[2].set_title('Attendance Percentage Distribution', fontsize=14)
axes[2].set_xlabel('Attendance (%)')
axes[2].set_ylabel('Count')
plt.tight_layout()
plt.show()

# --- Scatterplot with Employee Labels ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Tasks_Completed', y='Sales', hue='Rating', palette='viridis', s=150, edgecolor='black')
plt.title("Sales vs Tasks Completed (Colored by Performance Rating)", fontsize=16)
plt.xlabel("Tasks Completed", fontsize=14)
plt.ylabel("Sales ($)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
for _, row in df.iterrows():
    plt.text(row['Tasks_Completed']+0.3, row['Sales']+150, row['Name'], fontsize=9)
plt.legend(title='Performance Rating', bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

# --- Attendance Barplot ---
plt.figure(figsize=(12, 6))
bar = sns.barplot(data=df, x='Name', y='Attendance (%)', palette='coolwarm')
plt.title("Attendance Percentage by Employee", fontsize=16)
plt.ylabel("Attendance (%)", fontsize=14)
plt.xlabel("Employee Name", fontsize=14)
plt.ylim(80, 100)
plt.xticks(rotation=45)
for p in bar.patches:
    height = p.get_height()
    bar.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()

# --- Boxplot Sales by Rating ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Rating', y='Sales', palette='Set2', width=0.6)
sns.stripplot(data=df, x='Rating', y='Sales', color='black', alpha=0.6, jitter=True, size=7)
plt.title("Sales by Performance Rating", fontsize=16)
plt.xlabel("Performance Rating", fontsize=14)
plt.ylabel("Sales ($)", fontsize=14)
plt.show()

# --- Performance Score ---
df['Performance_Score'] = (df['Sales'] * 0.4) + (df['Tasks_Completed'] * 0.3) + (df['Attendance (%)'] * 0.3)
print("\nTop 5 Employees by Overall Performance Score:")
print(df[['Name', 'Performance_Score']].sort_values(by='Performance_Score', ascending=False).head())

# --- KMeans Clustering ---
features = df[['Sales', 'Tasks_Completed', 'Attendance (%)']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Tasks_Completed', y='Sales', hue='Cluster', palette='Set1', s=100)
plt.title("Employee Clusters Based on Performance", fontsize=16)
plt.xlabel("Tasks Completed")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# --- KPI Summary ---
top_sales = df['Sales'].max()
top_employee = df.loc[df['Sales'].idxmax(), 'Name']
avg_tasks = df['Tasks_Completed'].mean()
avg_attendance = df['Attendance (%)'].mean()
print(f"\nKPI Summary:")
print(f"Top Sales: ${top_sales} by {top_employee}")
print(f"Average Tasks Completed: {avg_tasks:.2f}")
print(f"Average Attendance: {avg_attendance:.2f}%")