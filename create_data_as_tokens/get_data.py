import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

print("Starting")


def start_by_combinning():
    cgm_df = pd.read_csv('CGM_train.csv')
    # keep only thos whos RegistrationCode starts with 10K
    cgm_df = cgm_df[cgm_df['RegistrationCode'].str.startswith('10K')]
    sqlog = pd.read_csv('SQLog_train.csv')
    sqlog.reset_index(inplace=True)
    log = sqlog[sqlog.RegistrationCode.str.startswith('10K')]
    # Remove rows with negative values as all values should be positive
    idx = ((log.drop(['RegistrationCode', 'Date', 'meal_type', 'score'], axis=1) >= 0).all(axis=1))
    log = log[idx]
    log['Date'] = pd.to_datetime(log['Date'])
    cgm_df['Date'] = pd.to_datetime(cgm_df['Date'])
    cgm_df.sort_values(by=['Date'], inplace=True)
    log.sort_values(by=['Date'], inplace=True)
    data = pd.merge_asof(cgm_df, log, on='Date', by='RegistrationCode', tolerance=pd.Timedelta("899s"),
                         direction='nearest')
    data.sort_values(by=['RegistrationCode', 'Date'], inplace=True)
    data.index = data['RegistrationCode']

    return data


data = start_by_combinning()

general_cols = ['GlucoseValue', 'PPGR']

nutritional_cols = ['energy_kcal',
                    'carbohydrate_g',
                    'protein_g',
                    'caffeine_mg',
                    'water_g',
                    'totallipid_g',
                    'alcohol_g',
                    'sugarstotal_g',
                    'cholesterol_mg']

time = ['day_of_week', 'hour', 'minute', 'month', 'Date']

general_cols += nutritional_cols + time


def clip_by_q(data):
    for col in nutritional_cols:
        print(f"clipping {col}")
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].hist(data[col], bins=100, log=True)
        axs[0].set_title(f'{col} before clipping')
        axs[0].set_xlabel(col)
        axs[0].set_ylabel('Frequency')
        data[col] = data[col].clip(0, data[data[col] > 0][col].quantile(0.99))
        axs[1].hist(data[col], bins=100, log=True)
        axs[1].set_title(f'{col} after clipping')
        axs[1].set_xlabel(col)
        axs[1].set_ylabel('Frequency')
        plt.show()

    return data


data = clip_by_q(data)

names = data.index.unique()
per_person = []
for i, name in enumerate(tqdm(names)):
    per_person.append(data.loc[name])


def re_org_time(per_person):
    cgm_diet = per_person.copy()
    cgm_diet2 = []
    names = []
    for i in tqdm(range(len(cgm_diet))):
        data = cgm_diet[i]
        data.index = data["RegistrationCode"]
        data = data.drop("RegistrationCode", 1)
        data = data.drop("ConnectionID", 1)
        data["Date"] = data["Date"].apply(lambda x: str(x))
        day_of_week = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                       'Sunday': 6}
        data["day_of_week"] = data["Date"].apply(
            lambda x: day_of_week[datetime.datetime.strptime(x.split()[0], '%Y-%m-%d').strftime('%A')])
        # split Date column into date and hour
        data["date"] = data["Date"].apply(lambda x: x.split()[0])
        data["hours"] = data["Date"].apply(lambda x: x.split()[1])
        data["hour"] = data["hours"].apply(lambda x: int(x.split(":")[0]))
        data["minute"] = data["hours"].apply(lambda x: int(x.split(":")[1]))
        data['month'] = data["date"].apply(lambda x: int(x.split("-")[1]))
        data['year'] = data["date"].apply(lambda x: int(x.split("-")[0]))
        # and make them into a single numerical value
        # data = data.drop("Date", 1)
        data = data.drop("hours", 1)
        data = data.drop("date", 1)

        # clip GlucoseValue to 40 - 500
        data['GlucoseValue'] = data['GlucoseValue'].clip(40, 500)
        # starch_g  glucose_g   sucrose_g  totaltransfattyacids_g  starch_g  score
        data = data[general_cols]

        # if the sum of col energy_kcal is less than 1000 skip this participant
        if data['energy_kcal'].sum() < 1000:
            print(f"skipping {i} because sum of energy_kcal is less than 1000")
            continue
        cgm_diet[i] = data

        # impute missing values with 0
        data = data.fillna(0)
        data['Date'] = pd.to_datetime(data['Date'])
        # if difference between each row is more than 15 minutes
        if data['Date'].diff().dt.total_seconds().gt(905).any():
            print(f"splitting {i}")
            # if the difference is more than an hour we do nothing (it will get split later)
            # but if its less than an hour, we want to add places for the missing values with nans
            # check for places where the difference is more than 15 minutes but less than an hour
            jumps = data['Date'].diff().dt.total_seconds().gt(1400) & data['Date'].diff().dt.total_seconds().lt(
                60 * 60 * 4)
            # for each we want to add nans (as many times such that the is exactly 15 minutes between each row)
            for i in data[jumps]['Date']:
                # i is the index (name) of the row where the jump is
                # find the difference in seconds
                times = pd.DataFrame(data.index, index=data['Date'])
                diff = i - times.index[times.index.get_loc(i) - 1]
                diff = diff.total_seconds()
                # find the number of nans to add
                nans_to_add = round((diff / 900)) - 1
                # create a df with nans starting at the time before i in temp_df + 15 minutes
                # time to start is the time in temp_df just before i. so lets look at the index before i in temp_df
                start_time = times.index[times.index.get_loc(i) - 1] + pd.Timedelta("15m")
                nan_df = pd.DataFrame(np.nan, index=pd.date_range(start=start_time, periods=nans_to_add, freq='15T'),
                                      columns=data.columns)
                # make the index the same as the data, and move the index to Date column
                nan_df['Date'] = nan_df.index
                nan_df.index = data.index[:len(nan_df)]
                # add the nan_df to temp_df
                cut_off = times.index.get_loc(i)
                temp_df = pd.concat([data.iloc[:cut_off], nan_df, data.iloc[cut_off:]])

            temp_df['GlucoseValue'] = temp_df['GlucoseValue'].interpolate(method='linear', limit_direction='both')
            temp_df['PPGR'] = temp_df['PPGR'].interpolate(method='linear', limit_direction='both')
            temp_df['day_of_week'] = temp_df['day_of_week'].interpolate(method='linear', limit_direction='both')
            temp_df['hour'] = temp_df['hour'].interpolate(method='linear', limit_direction='both')
            temp_df['minute'] = temp_df['minute'].interpolate(method='linear', limit_direction='both')
            temp_df['month'] = temp_df['month'].interpolate(method='linear', limit_direction='both')
            # temp_df['year'] = temp_df['year'].interpolate(method='linear', limit_direction='both')

            data = temp_df

        cgm_diet2.append(data)
        name = data.index[0][4:]
        names.append(name)

    return cgm_diet, cgm_diet2, names


cgm_diet, cgm_diet2, names = re_org_time(per_person)

# print how many we lost from cgm_diet and cgm_diet2
print(f" here we removed participants if the sum of energy_kcal was less than 1000")
print(f" We had {len(cgm_diet)} participants, and now we have {len(cgm_diet2)} participants")


def re_org_diet_thresholds(cgm_diet2):
    cgm_diet = cgm_diet2.copy()
    cgm_diet_filtered = []
    for i in tqdm(range(len(cgm_diet))):
        data = cgm_diet[i]
        # Assuming 'Date' column is already in datetime format; if not, convert it
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = data['Date'].dt.date

        # Sum energy_kcal per day
        daily_sum = data.groupby(['Day'])['energy_kcal'].sum().reset_index()

        # Filter days with more than 500 calories and less than 7000
        high_energy_days = daily_sum[daily_sum['energy_kcal'] > 500]['Day']
        low_energy_days = daily_sum[daily_sum['energy_kcal'] < 7000]['Day']

        # days where there are at least 3 logs - i.e 3 non zero values
        counted_3_meals = data.groupby(['Day'])['energy_kcal'].apply(lambda x: (x > 0).sum())
        counted_3_meals = counted_3_meals[counted_3_meals >= 3].index

        # Filter the original data to include only rows from high energy days
        filtered_data = data[data['Day'].isin(high_energy_days)]
        filtered_data = filtered_data[filtered_data['Day'].isin(counted_3_meals)]
        filtered_data = filtered_data[filtered_data['Day'].isin(low_energy_days)]

        # filter only days that are not the first or last day
        filtered_data = filtered_data[filtered_data['Day'] != filtered_data['Day'].min()]
        filtered_data = filtered_data[filtered_data['Day'] != filtered_data['Day'].max()]

        # If there is data remaining after the filter, add it to the new list
        if not filtered_data.empty:
            cgm_diet_filtered.append(filtered_data)

    return cgm_diet_filtered


cgm_diet_filtered = re_org_diet_thresholds(cgm_diet2)

# print how many we lost from cgm_diet and cgm_diet_filtered
print(
    f" here we removed days that had less than 300 calories or over 7000, or less than 3 logs, and the first and last day")
print(f" We had {len(cgm_diet)} participants, and now we have {len(cgm_diet_filtered)} participants")
# calculate how many days we had before and after filtering
unique_days_per_participant = sum([len(df['Day'].unique()) for df in cgm_diet])
unique_days_per_participant_filtered = sum([len(df['Day'].unique()) for df in cgm_diet_filtered])
print(f" We had {unique_days_per_participant} days, and now we have {unique_days_per_participant_filtered} days")


def plot_cal_in():
    # PLOTTING CODE:
    # Create a DataFrame to hold all participants' data with the date and caloric intake
    all_participants_data = pd.DataFrame()
    for i in tqdm(range(len(cgm_diet_filtered))):
        # Extract the current participant's data
        data = cgm_diet_filtered[i]

        # Convert the Date column from string to datetime, if not already done
        data['Date'] = pd.to_datetime(data['Date'])

        # Strip the time component and keep only the date for grouping
        data['Day'] = data['Date'].dt.date

        # Group by participant name and day, then sum the energy_kcal
        daily_caloric_intake = data.groupby([data.index, 'Day'])['energy_kcal'].sum().reset_index()
        daily_caloric_intake.rename(columns={'energy_kcal': 'Total Daily Caloric Intake'}, inplace=True)

        # Append the result to the all_participants_data DataFrame
        all_participants_data = pd.concat([all_participants_data, daily_caloric_intake], ignore_index=True)
    # Set the index to participant name and day
    all_participants_data.set_index(['RegistrationCode', 'Day'], inplace=True)
    # plot histogram of caloric intake
    all_participants_data['Total Daily Caloric Intake'].hist(bins=100, log=True)
    plt.xlabel('Total Daily Caloric Intake')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Daily Caloric Intake')
    plt.show()


plot_cal_in()

# find the df with the most columns in data
cols = []
for df in cgm_diet_filtered:
    if len(df.columns) > len(cols):
        cols = df.columns

print(cols)

# make sure that all the dataframes have all columns (those who dont, add them all as 0)
# also make sure the cols are in the same order
for i, df in tqdm(enumerate(cgm_diet_filtered)):
    for col in cols:
        if col not in cgm_diet_filtered[i].columns:
            cgm_diet_filtered[i][col] = 0
    cgm_diet_filtered[i] = cgm_diet_filtered[i][cols]


def comb_close_meals(cgm_diet_filtered):
    cgm_diet_filtered_diet_processed = cgm_diet_filtered.copy()
    # go over all of the data, and if you find two or more diet logging (you can identefy by energy_kcal >0)
    # whithin a window, combine them into one diet logging (sum the values, replace the first with the sum, and zero the second)
    for i in tqdm(range(len(cgm_diet_filtered_diet_processed))):
        data = cgm_diet_filtered_diet_processed[i].copy()
        # alter index such that it wll be unique, and we can pull out the data for each participant
        # since at the moment the index is the same for all participants (the RegistrationCode)
        data.index = [j for j in range(len(data))]
        l = len(nutritional_cols) + 2
        # find the times where the energy_kcal is around 400 - 700
        # create a temporary col with the sum of the nutritional cols (2 - l)
        data['sum'] = data.iloc[:, 2:l].sum(axis=1)
        idx = data['sum'].gt(0)
        # remove the temporary col
        data = data.drop('sum', 1)
        for j in idx[idx].index:
            if j + 3 >= len(data):
                continue
            if idx[j + 1] and idx[j + 2] and idx[j + 3]:
                data.iloc[j, 2:l] = data.iloc[j, 2:l] + \
                                    data.iloc[j + 1, 2:l] + \
                                    data.iloc[j + 2, 2:l] + \
                                    data.iloc[j + 3, 2:l]
                data.iloc[j + 1, 2:l] = 0
                data.iloc[j + 2, 2:l] = 0
                data.iloc[j + 3, 2:l] = 0
            elif idx[j + 1] and idx[j + 2]:
                data.iloc[j, 2:l] = data.iloc[j, 2:l] + \
                                    data.iloc[j + 1, 2:l] + \
                                    data.iloc[j + 2, 2:l]
                data.iloc[j + 1, 2:l] = 0
                data.iloc[j + 2, 2:l] = 0
            elif idx[j + 1]:
                data.iloc[j, 2:l] = data.iloc[j, 2:l] + \
                                    data.iloc[j + 1, 2:l]
                data.iloc[j + 1, 2:l] = 0
        cgm_diet_filtered_diet_processed[i] = data

    return cgm_diet_filtered_diet_processed


cgm_diet_filtered_diet_processed = comb_close_meals(cgm_diet_filtered)
l = len(nutritional_cols) + 2

for i in tqdm(range(len(cgm_diet_filtered_diet_processed))):
    cgm_diet_filtered_diet_processed[i].index = cgm_diet_filtered[i].index


def align(cgm_diet_filtered_diet_processed):
    cgm_diet_filtered_diet_processed_aligned = cgm_diet_filtered_diet_processed.copy()
    for i in tqdm(range(len(cgm_diet_filtered_diet_processed_aligned))):
        data = cgm_diet_filtered_diet_processed_aligned[i].copy()
        data.reset_index(drop=True, inplace=True)

        # Find the times where the total sugar is above 10
        idx = data['sugarstotal_g'].gt(10)
        # or calories are above 500
        idx2 = data['energy_kcal'].gt(100)

        idx = idx | idx2

        # Get the glucose value at that time, and 1 hour later (4*15 minutes)
        for j in idx[idx].index:
            glucose = data.loc[j, 'GlucoseValue']
            if j + 4 >= len(data):
                continue
            glucose2 = data.loc[j + 4, 'GlucoseValue']

            # Get the difference
            diff = glucose2 - glucose

            # Look for a window of 1 hours around the time of the first glucose value
            window_start = max(0, j - 3)
            window_end = min(len(data), j + 3)
            window = data.loc[window_start:window_end]

            # Find the min and max values in the window
            min_val = window['GlucoseValue'].min()
            max_val = window['GlucoseValue'].max()

            # If within an hour of eating we finder a larger spike, or no spike at all, move the diet logging
            if (max_val - min_val) > diff + 10 or diff < 0:
                # Locate the spike in the window
                spike = window[window['GlucoseValue'] == max_val].iloc[0]

                # Move the diet logging to an hour before the spike
                spike_idx = spike.name  # Get the index of the spike in the data
                new_idx = spike_idx - 4  # 1 hour before the spike

                if new_idx >= 0:
                    # Move the diet logging (cols 2 to 23) to an hour before the spike
                    data.iloc[new_idx, 2:l] += data.iloc[j, 2:l]
                    # Set the diet logging at j to 0
                    data.iloc[j, 2:l] = 0
        cgm_diet_filtered_diet_processed_aligned[i] = data

    return cgm_diet_filtered_diet_processed_aligned


cgm_diet_filtered_diet_processed_aligned = align(cgm_diet_filtered_diet_processed)

for i in tqdm(range(len(cgm_diet_filtered_diet_processed_aligned))):
    cgm_diet_filtered_diet_processed_aligned[i].index = cgm_diet_filtered[i].index

data = cgm_diet_filtered_diet_processed_aligned
print("data gotten")


def drop_duplicate_columns(df):
    df_transpose = df.T.drop_duplicates().T
    return df_transpose


# Preprocess each DataFrame to drop duplicate columns
data = [drop_duplicate_columns(df) for df in tqdm(data)]

# find the df with the most columns in data
cols = []
for df in data:
    if len(df.columns) > len(cols):
        cols = df.columns

print(cols)

# make sure that all the dataframes have all columns (those who dont, add them all as 0)
# also make sure the cols are in the same order
for i, df in tqdm(enumerate(data)):
    for col in cols:
        if col not in data[i].columns:
            data[i][col] = 0
    data[i] = data[i][cols]

# Assuming 'GlucoseValue' is the first glucose measurement column
# Nutrient columns might need adjusting depending on the actual data
nutrient_columns = data[0].columns[2:l - 1]  # Adjust indices appropriately

# Aggregate all data for each nutrient column to calculate quantiles
combined_data = {col: pd.concat([df[col] for df in data if col in df.columns]) for col in tqdm(nutrient_columns)}

#####################
#   BINNING STAGE   #
#####################

# Calculate global bins for each nutrient based on combined data
bins = {}
for col, values in tqdm(combined_data.items()):
    non_zero_values = values[values > 0].dropna()
    if not non_zero_values.empty:
        unique_bins = np.quantile(non_zero_values, np.linspace(0, 1, 16))
        bins[col] = sorted(set(unique_bins))
        if len(bins[col]) < 2:
            bins[col].append(bins[col][0] + 1)  # Ensure at least two bins
        # print per nutrient its bin edges
        print(f"Nutrient: {col} - Bins: {bins[col]}")

# use these precalculated bins (came form the code above form train set:
nut_bins = {
    'energy_kcal': [0.02, 31.68, 84.52, 146.04, 195.00, 262.12, 340.58, 423.10, 518.50, 623.53, 745.56, 891.08, 1069.25,
                    1317.02, 1726.57, 5832.24],

    'carbohydrate_g': [0.00, 3.47, 9.05, 15.00, 21.60, 28.87, 36.13, 43.08, 51.81, 61.97, 73.86, 88.02, 106.90, 132.38,
                       177.13, 608.09],

    'protein_g': [0.00, 0.66, 1.63, 2.81, 4.42, 6.80, 9.61, 13.64, 18.23, 23.76, 30.46, 38.61, 49.34, 65.22, 93.21,
                  354.53],

    'caffeine_mg': [0.30, 188.52, 377.04, 754.08, 2639.28],

    'water_g': [0.00, 4.56, 36.15, 78.24, 129.46, 182.26, 247.63, 311.44, 389.16, 468.20, 498.60, 575.58, 687.39,
                855.37,
                1109.35, 4507.10],

    'totallipid_g': [0.00, 0.41, 0.89, 2.59, 5.68, 8.74, 11.97, 15.84, 19.97, 24.95, 30.71, 37.91, 47.38, 60.38, 83.04,
                     306.73],

    'alcohol_g': [0.02, 0.88, 2.60, 6.60, 9.90, 13.20, 18.00, 20.00, 26.40, 28.80, 33.00, 49.50, 66.00, 172.84],

    'sugarstotal_g': [0.00, 1.25, 2.82, 4.67, 6.88, 9.27, 12.12, 14.70, 18.02, 22.04, 26.04, 31.28, 37.90, 48.22, 65.61,
                      266.78],
}
# Apply bins to each DataFrame
for df in tqdm(data):
    for col in bins:
        if col in df.columns:
            df[col] = pd.cut(df[col], bins=nut_bins[col], labels=False, include_lowest=True, duplicates='drop')


# Additional function to extract and repeat time data based on token expansions
def extract_time_data(df):
    time_columns = df.columns[-6:]  # Assuming the last 7 columns are time-related
    time_data = df[time_columns]
    return time_data


def expand_time_data(time_data, tokens, modality_indicator):
    expanded_time_data = []
    token_idx = 0  # Keep track of the position in tokens

    # Iterate over each row in the original DataFrame
    for index, row in time_data.iterrows():
        # Count how many times this row's time data should be repeated
        # This should match the number of tokens derived from this particular time row
        num_tokens_for_this_row = 1  # Start with 1 for the glucose value itself
        while token_idx + num_tokens_for_this_row < len(modality_indicator) and \
                modality_indicator[token_idx + num_tokens_for_this_row] != 0:
            num_tokens_for_this_row += 1  # Include nutrients associated with this glucose measurement

        # Now repeat the time row this many times
        for _ in range(num_tokens_for_this_row):
            expanded_time_data.append(row.values.tolist())

        token_idx += num_tokens_for_this_row  # Move the token index forward by the number of tokens processed

    return pd.DataFrame(expanded_time_data, columns=time_data.columns)


#####################
# TOKENIZING STAGE  #
#####################

# Process for creating lists
tokens_list = []
modality_indicators_list = []
time_reg = []
time_expanded = []

for df in tqdm(data):
    tokens = []
    modality_indicator = []  # 0 for glucose, 1-N for each nutrient

    # Time data extraction
    regular_time_data = extract_time_data(df)
    time_reg.append(regular_time_data)

    # Iterate over DataFrame rows
    for index, row in df.iterrows():
        glucose_value = int(row['GlucoseValue'])
        tokens.append(glucose_value)
        modality_indicator.append(0)  # Glucose modality

        # Check each nutrient in the filtered columns
        for i, col_name in enumerate(bins.keys()):
            if col_name in df.columns:
                nutrient_value = row[col_name]
                if nutrient_value > 0:
                    tokens.append(-int(nutrient_value))  # Use negative values to indicate nutrients
                    modality_indicator.append(i + 1)  # Incrementing modality index for each nutrient

    tokens_list.append(np.array(tokens))
    modality_indicators_list.append(np.array(modality_indicator))
    expanded_time = expand_time_data(regular_time_data, tokens, np.array(modality_indicator))
    time_expanded.append(expanded_time)

time_reg_list, time_expanded_list = time_reg, time_expanded
# find in time_reg_list a df with nan
for i, time_reg in enumerate(time_reg_list):
    if time_reg.isnull().values.any():
        print(i)
        print(f" row with nan: {time_reg[time_reg.isnull().any(axis=1)]}")
        raise ValueError("stop ___")

# for time lists take only first 5 columns since they are numerical
time_reg_list = [time_reg.iloc[:, :4].values.astype(int) for time_reg in time_reg_list]
time_expanded_list = [time_expanded.iloc[:, :4].values.astype(int) for time_expanded in time_expanded_list]

# col minute (3rd column) set the value to be the closest to either (0, 15, 30, 45)
for time_reg in time_reg_list:
    time_reg[:, 2] = np.round(time_reg[:, 2] / 15) * 15
for time_expanded in time_expanded_list:
    time_expanded[:, 2] = np.round(time_expanded[:, 2] / 15) * 15

# number of unique values in the modality indicators - lets calculate it by looking how many unique values are in the modality indicators
number_of_modalities = {}
for modality_indicators in modality_indicators_list:
    for modality in modality_indicators:
        number_of_modalities[modality] = 1
number_of_modalities = len(number_of_modalities)
print(f"Number of modalities: {number_of_modalities}")

# for each modality indicator, we will raise its corresponding value in tokens_list by 100 * modality_indicator
# so modality 0 will remain as is. modality 1 will be raised by 100, modality 2 by 200, etc.
# we will also take the absolute value of the tokens, before doing this operation

for tokens, modality_indicators in zip(tokens_list, modality_indicators_list):
    for i, modality in enumerate(modality_indicators):
        if modality > 0:
            tokens[i] = abs(tokens[i]) + 500 + (16 * (modality - 1))

# I want to know the distribution of the tokens, so I will plot a histogram of the tokens
flattened_tokens = np.concatenate(tokens_list)
unique_values_in_tokens = np.unique(flattened_tokens)

# take off the min value from the tokens
min_value = unique_values_in_tokens[0] - 1
for tokens in tokens_list:
    tokens -= min_value
vocab_size = max(flattened_tokens) - min_value + 1
print(f"Vocab size: {vocab_size}")

# 1. make into integers
for i in range(len(time_expanded_list)):
    time_expanded_list[i] = time_expanded_list[i].astype(int)

# 2. divide minute by 15
for i in range(len(time_expanded_list)):
    time_expanded_list[i][:, 2] = time_expanded_list[i][:, 2] // 15

# 3. subtract the min value
concat_all = np.concatenate(time_expanded_list)

# all unique values per column in concat_all
unique_values_per_column = [np.unique(concat_all[:, i]) for i in range(concat_all.shape[1])]

# min value per column
min_value = np.min(concat_all, axis=0)
# add 1 to last 2 values
min_value[-2:] = 1
for i in range(len(time_expanded_list)):
    time_expanded_list[i] -= min_value
# clip them to 0 if negative
for i in range(len(time_expanded_list)):
    time_expanded_list[i] = np.clip(time_expanded_list[i], 0, None)

# show distribution of values per col
# all unique values per column in concat_all
concat_all = np.concatenate(time_expanded_list)
unique_values_per_column = [np.unique(concat_all[:, i]) for i in range(concat_all.shape[1])]
for i, unique_values in enumerate(unique_values_per_column):
    plt.figure(figsize=(20, 10))
    plt.hist(concat_all[:, i], bins=len(unique_values), log=True)
    plt.show()

# for each
# 4. calc how many unique values are in each column
concat_all = np.concatenate(time_expanded_list)
temporal_vocab_size = [len(np.unique(concat_all[:, i])) for i in range(concat_all.shape[1])]
print(f"Temporal vocab size: {temporal_vocab_size}")

# Find the maximum length among all arrays
max_length = max(max(len(tokens) for tokens in tokens_list),
                 max(len(modalities) for modalities in modality_indicators_list))
# Pad each array to this maximum length
padded_tokens_list = [np.pad(tokens, (0, max_length - len(tokens)), mode='constant', constant_values=0) for tokens
                      in tokens_list]
padded_modality_indicators_list = [
    np.pad(modalities, (0, max_length - len(modalities)), mode='constant', constant_values=number_of_modalities) for
    modalities in modality_indicators_list]

padded_time_expanded_list = [np.pad(time_expanded, ((0, max_length - time_expanded.shape[0]), (0, 0)),
                                    mode='constant', constant_values=0) for time_expanded in time_expanded_list]

# Convert these lists to PyTorch tensors
padded_tokens_tensor = torch.tensor(padded_tokens_list, dtype=torch.long)
padded_modality_indicators_tensor = torch.tensor(padded_modality_indicators_list, dtype=torch.long)
padded_time_expanded_tensor = torch.tensor(padded_time_expanded_list, dtype=torch.long)

temporal_vocab_size = [i + 1 for i in temporal_vocab_size]

torch.save({"tokens": padded_tokens_tensor,
            "modalities": padded_modality_indicators_tensor,
            "time_expanded": padded_time_expanded_tensor,
            "vocab_size": vocab_size,
            "temporal_vocab_size": temporal_vocab_size,
            "number_of_modalities": number_of_modalities,
            "index": [int(d.index[0][4:]) for d in cgm_diet_filtered], },
           "./cgm_diet_filtered_processed_aligned_tokenized_tensors_train.pt")
