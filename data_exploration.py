import pandas as pd


class DataLoader:
    """
    DataLoader class for loading data from a CSV file.
    """

    def __init__(self, filepath):
        """
        Initialize DataLoader with the file path of the data file.

        Parameters:
        filepath (str): The file path of the dataset.
        """
        self.filepath = filepath

    def load_data(self):
        """
        Load data from the CSV file.

        Returns:
        DataFrame: The loaded data as a pandas DataFrame.
        """
        try:
            data = pd.read_csv(self.filepath)
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


class DataSummary:
    """
    DataSummary class for providing a summary of the dataset.
    """

    def __init__(self, data):
        """
        Initialize DataSummary with data.

        Parameters:
        data (DataFrame): The dataset to summarize.
        """
        self.data = data

    def summary(self):
        """
        Generate a summary of the dataset.

        Returns:
        DataFrame: Summary of the dataset including descriptive statistics.
        """
        return self.data.describe(include="all")


class DataCleaning:
    """
    DataCleaning class for handling missing values and checking data types.
    """

    def __init__(self, data):
        """
        Initialize DataCleaning with data.

        Parameters:
        data (DataFrame): The dataset to clean.
        """
        self.data = data

    def check_data_types(self):
        """
        Check the data types of the columns in the dataset.

        Returns:
        Series: Data types of the columns.
        """
        return self.data.dtypes

    def check_missing_values(self):
        """
        Check for missing values in the dataset.

        Returns:
        Series: Counts of missing values in each column.
        """
        return self.data.isnull().sum()

    def fill_missing_values(self, fill_strategy="median"):
        """
        Fill missing values in the dataset.

        Parameters:
        fill_strategy (str): The strategy to use for filling missing values ('median' or 'mean').
        """
        for column in self.data.select_dtypes(include="number").columns:
            if fill_strategy == "median":
                self.data[column].fillna(self.data[column].median(), inplace=True)
            elif fill_strategy == "mean":
                self.data[column].fillna(self.data[column].mean(), inplace=True)
        print("Missing values filled.")
