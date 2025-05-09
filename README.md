# Forecasting-house-prices-accurately-using-smart-regression-techniques-in-data-science
# House Filtering Project

This project is a Python application that helps filter house listings based on user input.  
The user can enter the number of bedrooms, bathrooms, and floors, and the program will display matching houses from a dataset.

## Features
- Takes user input for:
  - Number of Bedrooms
  - Number of Bathrooms
  - Number of Floors
- Filters and displays matching house details from an Excel file (`house.xlsx`).

## How to Run
1. Make sure you have **Python** installed.
2. Install **pandas** and **openpyxl** and **gradio** if not already installed:
   ```bash
   pip install pandas openpyxl
   pip install gardio or !pip install gardio
   import pandas as pd

# Read the excel file into a pandas dataframe.
# Note that content/dataset (1).xlsx should be the actual path to your excel file.
df = pd.read_excel('/content/dataset (1).xlsx') 

# Display the dataframe.
display(df)

  Awesome! Here's a simple and clean README.md you can use:

# House Filtering Project

This project is a Python application that helps filter house listings based on user input.  
The user can enter the number of bedrooms, bathrooms, and floors, and the program will display matching houses from a dataset.

## Features
- Takes user input for:
  - Number of Bedrooms
  - Number of Bathrooms
  - Number of Floors
- Filters and displays matching house details from an Excel file (`house.xlsx`).

## How to Run
1. Make sure you have **Python** installed.
2. Install **pandas** and **openpyxl** if not already installed:
   ```bash
   pip install pandas openpyxl

3. Run the script:

python your_script_name.py



Files

house.xlsx — Dataset containing house information

your_script_name.py — Python script for filtering houses


Requirements

Python 3.x

pandas

openpyxl


Example

Enter number of bedrooms: 3
Enter number of bathrooms: 2
Enter number of floors: 5


Author
N.Logeshwaran
