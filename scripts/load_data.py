import pandas as pd

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

if __name__ == "__main__":
    data = load_dataset("Country's_life_expectancy_data(2016_2024).csv")
    print(data.head())
