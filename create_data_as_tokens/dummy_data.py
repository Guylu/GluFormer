import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Function to generate random dates
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# Define the number of rows and the date range
num_rows = 10  # Adjust as needed
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

# Create the DataFrame
data = {
    'tokens': np.random.randint(0, 400, size=num_rows),
    'time': [random_date(start_date, end_date) for _ in range(num_rows)]
}

df = pd.DataFrame(data)

# save the DataFrame to a CSV file
df.to_csv('dummy_data.csv', index=False)