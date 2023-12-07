import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class DataPreprocessor:
    """
    A class dedicated to preprocessing data for machine learning models.

    This class provides a comprehensive approach to preparing a dataset for model training and evaluation. It includes functionalities such as splitting the data, handling missing values, encoding categorical variables, and scaling numerical features. The class aims to streamline the preprocessing steps and ensure that the data is in the optimal format for model training.

    Methods:
    --------
    preprocess_data():
        Performs preprocessing steps on the dataset, including converting dates, encoding features, and splitting the data into training and testing sets.

    train_and_evaluate_model(X_train, y_train, preprocessor):
        Trains and evaluates the model using cross-validation, returning performance metrics.

    Attributes:
    -----------
    data : DataFrame
        The dataset to be preprocessed.

    Example:
    --------
    >>> preprocessor = DataPreprocessor(data)
    >>> X_train, X_test, y_train, y_test, pipeline_preprocessor = preprocessor.preprocess_data()
    >>> mean_mae, ci = preprocessor.train_and_evaluate_model(X_train, y_train, pipeline_preprocessor)
    """

    def __init__(self, data):
        """
        Initialize DataPreprocessor with the dataset.

        Parameters:
        data (DataFrame): The dataset to preprocess.
        """
        self.data = data

    def preprocess_data(self):
        """
        Preprocess the dataset by converting dates, encoding features, and setting up preprocessing steps.
        """
        # Convert 'Activity Period' to a datetime format and extract year and month
        self.data["Activity Period"] = pd.to_datetime(
            self.data["Activity Period"], format="%Y%m"
        )
        self.data["Year"] = self.data["Activity Period"].dt.year
        self.data["Month"] = self.data["Activity Period"].dt.month

        # Define the categorical and numerical features
        categorical_features = [
            "Operating Airline",
            "GEO Summary",
            "Activity Type Code",
        ]
        numerical_features = ["Year", "Month"]

        # Define the target variable
        target = "Passenger Count"
        X = self.data[numerical_features + categorical_features]
        y = self.data[target]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Preprocessing pipelines for numerical and categorical data
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Bundle preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return X_train, X_test, y_train, y_test, self.preprocessor

    def train_and_evaluate_model(self, X_train, y_train, preprocessor):
        """
        Train and evaluate the model using cross-validation.

        Parameters:
        X_train (DataFrame): The training data.
        y_train (Series): The training labels.
        preprocessor (ColumnTransformer): The preprocessor to use in the pipeline.
        """
        # Create a preprocessing and modeling pipeline
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=100, random_state=0)),
            ]
        )

        # Perform cross-validation
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )

        # Output the mean score and the confidence interval
        mean_score = -scores.mean()
        confidence_interval = scores.std() * 2
        return mean_score, confidence_interval


class FeatureEngineer:
    """
    A class for performing feature engineering tasks on a dataset.

    This class provides methods to enhance a dataset with additional features, such as polynomial features, which can help in improving the performance of machine learning models. It's particularly useful for datasets where relationships between variables are complex and not purely linear.

    Methods:
    --------
    add_polynomial_features(X_train, X_test, degree=2):
        Enhances the training and testing datasets with polynomial features of a specified degree.

    Parameters:
    -----------
    data : DataFrame
        The dataset on which feature engineering is to be performed.

    Example:
    --------
    >>> feature_engineer = FeatureEngineer(data)
    >>> X_train_enhanced, X_test_enhanced = feature_engineer.add_polynomial_features(X_train, X_test)
    """

    def __init__(self, data):
        self.data = data

    def add_polynomial_features(self, X_train, X_test, degree=2):
        """
        Add polynomial features to the training and testing sets.

        Parameters:
        X_train (DataFrame): The training data.
        X_test (DataFrame): The testing data.
        degree (int): The degree of the polynomial features.

        Returns:
        DataFrame, DataFrame: Enhanced training and testing sets with polynomial features.
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        numerical_features = [
            "Year",
            "Month",
        ]  # Specify the numerical features to be used

        X_train_poly = poly.fit_transform(X_train[numerical_features])
        X_test_poly = poly.transform(X_test[numerical_features])

        # Convert to DataFrame
        X_train_poly_df = pd.DataFrame(
            X_train_poly, columns=poly.get_feature_names_out(), index=X_train.index
        )
        X_test_poly_df = pd.DataFrame(
            X_test_poly, columns=poly.get_feature_names_out(), index=X_test.index
        )

        # Combine with original features
        X_train_enhanced = pd.concat(
            [X_train.drop(numerical_features, axis=1), X_train_poly_df], axis=1
        )
        X_test_enhanced = pd.concat(
            [X_test.drop(numerical_features, axis=1), X_test_poly_df], axis=1
        )

        return X_train_enhanced, X_test_enhanced


class ModelTrainer:
    """
    ModelTrainer class for training and evaluating machine learning models.

    This class is designed to facilitate the training of machine learning models with a focus on regression tasks. It provides methods for training models using grid search for hyperparameter optimization and for evaluating model performance. The class is flexible enough to be used with different types of regression models and datasets.

    Attributes:
    -----------
    preprocessor : ColumnTransformer
        A preprocessor object that transforms the dataset before feeding it into the model. This is typically a ColumnTransformer that handles both numerical and categorical preprocessing.

    Methods:
    --------
    train_with_grid_search(X_train, y_train, param_grid, cv_folds=3):
        Trains a RandomForestRegressor model using GridSearchCV to find the best hyperparameters. The method takes the training data, training labels, a parameter grid for hyperparameter tuning, and the number of cross-validation folds.

    train_linear_regression(X_train, y_train):
        Trains a simple Linear Regression model. The method takes the training data and training labels and fits a Linear Regression model to them.

    Parameters:
    -----------
    preprocessor : ColumnTransformer
        The preprocessor to apply to the data before training the model. This should be compatible with the data and the model being trained.

    Examples:
    ---------
    >>> from sklearn.compose import ColumnTransformer
    >>> preprocessor = ColumnTransformer(transformers=[...])
    >>> model_trainer = ModelTrainer(preprocessor)
    >>> best_model, best_params, best_score = model_trainer.train_with_grid_search(X_train, y_train, param_grid)
    >>> linear_model = model_trainer.train_linear_regression(X_train, y_train)
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def train_with_grid_search(self, X_train, y_train, param_grid, cv_folds=3):
        """
        Train a model using GridSearchCV.

        Parameters:
        X_train (DataFrame): Training data.
        y_train (Series): Training labels.
        param_grid (dict): Grid of parameters to search over.
        cv_folds (int): Number of cross-validation folds.

        Returns:
        Best model, best parameters, and best score.
        """
        pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("model", RandomForestRegressor(random_state=0)),
            ]
        )
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, scoring="neg_mean_absolute_error"
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        best_model = grid_search.best_estimator_

        return best_model, best_params, best_score

    def train_linear_regression(self, X_train, y_train):
        """
        Train a simple linear regression model.

        Parameters:
        X_train (DataFrame): Training data.
        y_train (Series): Training labels.

        Returns:
        Trained Linear Regression model.
        """
        linear_pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", LinearRegression())]
        )
        linear_pipeline.fit(X_train, y_train)
        return linear_pipeline
