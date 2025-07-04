import os
import pandas as pd

# Important paths
EXCEL_PATH = r"C:\Users\Public\citations.xlsx"
SHEET_NAME = "citations1"
OUTPUT_DIR = r"C:\Users\Public"
MD_PATH = os.path.join(OUTPUT_DIR, "LISTS_OF_CITATIONS.md")

# Categories mapped to Markdown sections
CATEGORIES = {
    "Books": "Books",
    "Conference Papers": "Conference Papers",
    "Data Articles": "Data Articles",
    "Doctoral Theses": "Doctoral Theses",
    "Final Graduation Projects": "Final Graduation Projects",
    "Journal Articles": "Journal Articles",
    "Master's Degree Dissertations": "Master's Degree Dissertations",
    "Other Articles": "Other Articles",
    "Repository Articles": "Repository Articles",
    "Specialization Monographs": "Specialization Monographs",
}

# Fixed header
HEADER = """
## Introduction

As far as we know, the 3W Dataset was useful and is cited by the {N} works listed in this document.

There is a dedicated section below for each type of work. These sections are presented in alphabetical order. In each section the works are listed according to the years in which they were published, from the most recent to the oldest.

## Our requests

If you know any other published work that cites the 3W Dataset, please let us know by commenting in [this](https://github.com/Petrobras/3W/discussions/3) discussion.

If you use any resource published in this Git repository, we ask that it be properly cited in your work. Click on the ***Cite this repository*** link on this repository landing page to access different citation formats supported by the GitHub citation feature.

## Lists of Citations

* [Books](#books)
* [Conference Papers](#conference-papers)
* [Data Articles](#data-articles)
* [Doctoral Theses](#doctoral-theses)
* [Final Graduation Projects](#final-graduation-projects)
* [Journal Articles](#journal-articles)
* [Master's Degree Dissertations](#masters-degree-dissertations)
* [Other Articles](#other-articles)
* [Repository Articles](#repository-articles)
* [Specialization Monographs](#specialization-monographs)
"""


def format_citation(row):
    """Formats a citation based on available columns."""
    columns = ["Author", "Title", "Institution/Event", "Year", "Link"]
    parts = [str(row[col]) for col in columns if pd.notna(row[col])]
    return ". ".join(parts) + "."


def process_excel_to_markdown():
    """Processes the Excel file and generates the Markdown file."""
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"The file 'citations.xlsx' was not found at {EXCEL_PATH}."
        )

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    required_columns = [
        "Author",
        "Title",
        "Institution/Event",
        "Category",
        "Year",
        "Link",
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"The file 'citations.xlsx' must contain the following columns: {', '.join(required_columns)}."
        )

    # Sort by year in descending order
    df = df.sort_values(by=["Year"], ascending=False)
    df["Formatted"] = df.apply(format_citation, axis=1)

    # Dictionary to store citations by category
    citations_by_category = {category: [] for category in CATEGORIES.values()}

    for _, row in df.iterrows():
        category = row["Category"]
        if category in CATEGORIES:
            citations_by_category[CATEGORIES[category]].append(f"1. {row['Formatted']}")

    # Constructing the final content
    citation_count = len(df)
    final_content = HEADER.replace("{N}", str(citation_count))

    for category, citations in citations_by_category.items():
        final_content += f"\n\n## {category}\n\n"
        final_content += "\n".join(citations) if citations else "1."

    # Saving the Markdown file
    with open(MD_PATH, "w", encoding="utf-8") as file:
        file.write(final_content)

    print(f"Updated Markdown file saved at: {MD_PATH}")


if __name__ == "__main__":
    process_excel_to_markdown()
