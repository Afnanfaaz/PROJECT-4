# __main__.py

from .data_preprocessing import DataPreprocessor, ModelTrainer, FeatureEngineer
from .data_exploration import DataCleaning, DataSummary, DataLoader
from .eda import EDAAnalyzer


def main():
    # Display a message to show that the package is being executed
    print("Executing the data analysis package.")

    # Demonstrating the instantiation of classes
    data_loader = DataLoader("path/to/your/data.csv")
    data_preprocessor = DataPreprocessor(None)
    eda_analyzer = EDAAnalyzer(None)

    # Here, we're not performing any actual data loading, preprocessing, or analysis
    # since the focus is on making the package installable and structurally correct
    print("Classes instantiated successfully.")


if __name__ == "__main__":
    main()
