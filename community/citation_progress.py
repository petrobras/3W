import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Plot style settings
mpl.style.use("default")
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=11)

# File paths
script_dir = os.path.dirname(__file__)  # 3W/community
xlsx_file = os.path.join(script_dir, "citations.xlsx")
output_file = os.path.join(script_dir, "progress_over_the_years.svg")
sheet_name = "citations1"

# Read data from Excel
df_raw = pd.read_excel(xlsx_file, sheet_name=sheet_name, usecols="E:F")
df_raw = df_raw.rename(columns={"Category": "Category", "Year": "Year"})

# Count publications by year and category
df_count = df_raw.groupby(["Year", "Category"]).size().unstack(fill_value=0)

# Reorder columns
ordered_columns = [
    "Books",
    "Conference Papers",
    "Data Articles",
    "Doctoral Theses",
    "Final Graduation Projects",
    "Journal Articles",
    "Master's Degree Dissertations",
    "Other Articles",
    "Repository Articles",
    "Specialization Monographs",
]

# Ensure all expected categories exist
for col in ordered_columns:
    if col not in df_count.columns:
        df_count[col] = 0

df_count = df_count[ordered_columns].sort_index()

# Compute cumulative sum
df_cumulative = df_count.cumsum()

# Plot the data
fig, ax = plt.subplots(figsize=(12, 9))
df_cumulative.plot(kind="bar", stacked=True, ax=ax, width=0.5)

# Axes and legend
plt.ylabel("Number of Citations")
plt.xlabel("Year")
plt.xticks(rotation=0)
ax.legend(title=None)

# Remove grid lines
plt.grid(False)

# Add totals on top of bars
totals = df_cumulative.sum(axis=1)
for i, total in enumerate(totals):
    ax.text(
        i,
        total + 0.5,
        str(int(total)),
        ha="center",
        va="bottom",
        fontsize=12,
        weight="bold",
        color="black",
    )

# Save figure
plt.savefig(output_file, format="svg", bbox_inches="tight")
plt.show()
