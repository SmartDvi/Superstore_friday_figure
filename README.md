# Comprehensive Business Analytics Dashboard

## Overview

The Comprehensive Business Analytics Dashboard is a sophisticated tool designed to deliver actionable insights and facilitate data-driven decision-making for businesses. This interactive dashboard features a series of dynamic pages, each focusing on different aspects of business performance:

- **Introduction Page**: Offers a detailed overview of the dashboard and its functionalities.
- **Data Overview Page**: Provides a summary of the dataset's structure and key metrics.
- **Product Page**: Analyzes product performance across categories and sub-categories.
- **Time Analysis Page**: Examines time-related trends in sales and profit.
- **Regression Page**: Implements regression analysis to forecast future outcomes.
- **Classification Page**: Utilizes classification techniques to categorize and interpret data.
- **Recommendation Page**: Generates recommendations based on historical data to guide strategic decisions.

## Features

- **Interactive Visualizations**: Create dynamic and engaging charts with Plotly.
- **Customizable Filters**: Filter data by region, city, category, and time periods.
- **Advanced Analytics**: Perform regression and classification analysis.
- **Actionable Recommendations**: Receive data-driven suggestions for business strategies.

## Installation

### Prerequisites

- Python 3.7 or later
- pip (Python package installer)

### Installation Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/SmartDvi/Bank-Marketing-Campaign.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd Bank-Marketing-Campaign
    ```

3. **Set Up a Virtual Environment:**

    ```bash
    python -m venv venv
    ```

4. **Activate the Virtual Environment:**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. **Install Required Packages:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Prepare the Dataset:**

    Ensure that the dataset `Sample_Superstore.xls` is located in the project directory. Update the file path in the code if necessary.

## Usage

1. **Launch the Application:**

    ```bash
    python main.py
    ```

2. **Access the Dashboard:**

    Open a web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

## Project Structure

- `main.py`: The entry point for running the Dash application.
- `assets/`: Contains CSS files for custom styling.
- `pages/`: Includes Python modules for each page of the dashboard:
  - `data_overview.py`: Data Overview Page
  - `product.py`: Product Page
  - `time_analysis.py`: Time Analysis Page
  - `regression.py`: Regression Page
  - `classification.py`: Classification Page
  - `recommendation.py`: Recommendation Page
  - `introduction.py`: Introduction Page
- `requirements.txt`: Lists the project's Python dependencies.
- `README.md`: This document.

## Contributing

Contributions to enhance this project are welcome. Please fork the repository and submit a pull request with your changes.

## Contact

For inquiries or support, please contact:

- **Moritus Peters** - [petersmoritus@gmail.com]

We appreciate your interest in the Comprehensive Business Analytics Dashboard and hope it provides valuable insights to drive your business forward.
