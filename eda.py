import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EDAAnalyzer:
    """
    EDAAnalyzer class for performing exploratory data analysis on the Air Traffic Passenger Statistics dataset.
    """

    def __init__(self, data):
        """
        Initialize EDAAnalyzer with the dataset.

        Parameters:
        data (DataFrame): The dataset to analyze.
        """
        self.data = data
        sns.set(style="whitegrid")  # Setting the plot style

    def plot_passenger_count_distribution(self):
        """
        Plot the distribution of Passenger Count.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data["Passenger Count"], bins=50, kde=True)
        plt.title("Distribution of Passenger Count")
        plt.xlabel("Passenger Count")
        plt.ylabel("Frequency")
        plt.show()

    def plot_flights_geo_region_distribution(self):
        """
        Plot the distribution of flights across GEO Regions.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x="GEO Region")
        plt.title("Distribution of Flights Across GEO Regions")
        plt.xlabel("GEO Region")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    def plot_flights_geo_summary_distribution(self):
        """
        Plot the distribution of flights by GEO Summary (Domestic vs. International).
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x="GEO Summary")
        plt.title("Distribution of Flights: Domestic vs. International")
        plt.xlabel("GEO Summary")
        plt.ylabel("Count")
        plt.show()

    def plot_yearly_passenger_counts(self):
        """
        Plot the total passenger counts aggregated by year.
        """
        # Ensure 'Activity Period' is in datetime format
        self.data["Activity Period"] = pd.to_datetime(
            self.data["Activity Period"], format="%Y%m"
        )
        self.data["Year"] = self.data["Activity Period"].dt.year
        yearly_passenger_counts = (
            self.data.groupby("Year")["Passenger Count"].sum().reset_index()
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=yearly_passenger_counts, x="Year", y="Passenger Count")
        plt.title("Yearly Passenger Counts Over Time")
        plt.xlabel("Year")
        plt.ylabel("Total Passenger Count")
        plt.show()

    def plot_passenger_counts_top_airlines(self):
        """
        Plot the passenger counts over time for the top 5 airlines.
        """
        top_airlines = self.data["Operating Airline"].value_counts().head(5).index
        top_airlines_data = self.data[self.data["Operating Airline"].isin(top_airlines)]
        yearly_airline_passenger_counts = (
            top_airlines_data.groupby(["Year", "Operating Airline"])["Passenger Count"]
            .sum()
            .reset_index()
        )

        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=yearly_airline_passenger_counts,
            x="Year",
            y="Passenger Count",
            hue="Operating Airline",
        )
        plt.title("Yearly Passenger Counts by Top 5 Airlines")
        plt.xlabel("Year")
        plt.ylabel("Total Passenger Count")
        plt.legend(title="Operating Airline")
        plt.show()

    def plot_passenger_counts_trend(self):
        """
        Plot the overall trend of passenger counts over time.
        """
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=self.data, x="Activity Period", y="Passenger Count")
        plt.title("Passenger Counts Over Time")
        plt.xlabel("Activity Period")
        plt.ylabel("Passenger Count")
        plt.show()
