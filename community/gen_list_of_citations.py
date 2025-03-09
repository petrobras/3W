"""This script processes an Excel file named 'citations.xlsx' containing citations
to the 3W Dataset and generates a Markdown file listing these citations.

The citations include relevant details such as authors, titles, institutions,
categories, years, and links, formatted in a consistent way. The resulting
Markdown file is saved in the specified output directory.

Note:
- The file 'citations.xlsx' must be located in the directory 'C:\\Users\\Public'.
- The sheet name within the Excel file must be 'citations1'.
- The file must include the following columns: 'Author', 'Title', 'Institution/Event',
  'Category', 'Year', and 'Link'.
"""

import os
import pandas as pd

# Important paths
#
EXCEL_PATH = r"C:\Users\Public\citations.xlsx"
SHEET_NAME = "citations1"
OUTPUT_DIR = r"C:\Users\Public"
MD_PATH = os.path.join(OUTPUT_DIR, "LIST_OF_CITATIONS.md")

# Fixed header for the Markdown file
#
HEADER = """
As far as we know, the 3W Dataset was useful and cited by the works listed below. If you know any other paper, final graduation project, master's degree dissertation or doctoral thesis that cites the 3W Dataset, we will be grateful if you let us know by commenting [this](https://github.com/Petrobras/3W/discussions/3) discussion. If you use any resource published in this repository, we ask that it be properly cited in your work. Click on the ***Cite this repository*** link on this repository landing page to access different citation formats supported by the GitHub citation feature.

This file (`LIST_OF_CITATIONS.md`) was generated automatically from records maintained in the `citations.xlsx` file.
"""


# Methods
#
def format_citation(row):
    """Formats a citation using non-empty columns from the row.

    Args:
        row (pd.Series): A row from the DataFrame containing citation details.

    Returns:
        str: A formatted citation string.
    """
    columns = ["Author", "Title", "Institution/Event", "Category", "Year", "Link"]
    parts = [str(row[col]) for col in columns if pd.notna(row[col])]
    return ". ".join(parts) + "."


def process_excel_to_markdown():
    """Processes the Excel file to generate a Markdown file with formatted citations.

    Raises:
        FileNotFoundError: If the Excel file is not found in the specified path.
        ValueError: If the required columns are not present in the Excel file.
    """
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"The file 'citations.xlsx' was not found in the directory "
            f"C:\\Users\\Public. Please ensure the file is placed in this directory and run the script again."
        )

    # Read the Excel file
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # Check for required columns
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
            f"The file 'citations.xlsx' must contain the following columns: "
            f"{', '.join(required_columns)}."
        )

    # Apply formatting to each row
    df["Formatted"] = df.apply(format_citation, axis=1)

    # Create a list of formatted citations
    formatted_citations = "\n\n".join(
        [f"1. {citation}" for citation in df["Formatted"]]
    )

    # Combine header and citations
    final_content = HEADER + formatted_citations

    # Ensure the output directory exists and write the Markdown file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as file:
        file.write(final_content)

    print(f"Updated Markdown file saved at: {MD_PATH}")


# Main execution
#
if __name__ == "__main__":
    process_excel_to_markdown()
