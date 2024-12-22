# StudentMarksAnalysis
This project provides a Streamlit-based interface for analyzing student marks datasets. It supports visualizations of both categorical and numeric features, and it accepts data in either CSV or Excel formats.

## Features
- **Categorical Features**: Displays countplots and violin plots for each categorical feature.
- **Numeric Features**: Displays histograms and boxplots for numeric features.
- **Pairplots**: Visualizes relationships between all features.
- Supports both CSV and Excel file formats.

## Requirements
To run this application, you need the following Python packages installed:

- `streamlit`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd <repository-directory>
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run student_marks_graphs_interface.py
   ```

5. Upload your dataset (CSV or Excel) using the provided file uploader in the web interface.

## File Structure
- **`student_marks_graphs_interface.py`**: The main Streamlit application script.
- **`README.md`**: This file, providing an overview of the project.

## Input Dataset Requirements
The uploaded dataset should:
- Contain a column named `Marks` (case-sensitive), which serves as the target variable.
- Include other features as either numeric or categorical columns.

## Example Datasets
Example datasets can be found in the `data/` directory (add your example datasets there).

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

