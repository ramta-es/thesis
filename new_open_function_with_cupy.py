import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# First just read the headers to get column names
col_names = pd.read_csv('/Volumes/Extreme Pro/20250605_121437_results_raw.csv', nrows=0).columns

# Filter column names
v_columns = [col for col in col_names if "[V]" in col]
nm_columns = [col for col in col_names if "[nm]" in col]

# Read the CSV only once with just the columns we need
df_selected1 = pd.read_csv('/Volumes/Extreme Pro/20250605_121437_results_raw.csv',
                         usecols=v_columns + nm_columns).to_numpy()
df_selected = pd.read_csv('/Volumes/Extreme Pro/20250605_121437_results_raw.csv',
                         usecols=v_columns + nm_columns).iloc[745:1953].to_numpy()


filtered = df_selected1[(df_selected1[:, 0] < 750) & (df_selected1[:, 0] > 400), :]
print('filtered shape',filtered.shape)
spectrometer_mat = df_selected[:, 1:]
wave_lengths = df_selected[:, 0]
percentage_mat = spectrometer_mat / np.sum(spectrometer_mat, axis=0)
# spectrometer_mat = (df_selected[v_columns].to_numpy())
# wave_lengths = df_selected[nm_columns].to_numpy()
print('df_selected shape: ', df_selected.shape)
print('wl shape', wave_lengths.shape)
print('spec mat shape', spectrometer_mat.shape)
print(np.sum((spectrometer_mat), axis=0).shape)

print(percentage_mat.shape)
print(np.sum(percentage_mat, axis=0))



if np.array_equal(df_selected[:1209, :], filtered):
    print("They are exactly equal.")
else:
    print("They do not match.")


'''
# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Line plot
ax1.plot(range(1209), np.sum(spectrometer_mat, axis=1), 'b-', label='sin(x)')
ax1.set_title('not normalized')
ax1.set_xlabel('X')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend()
ax2.plot(range(1209), np.sum(percentage_mat, axis=1), 'b-', label='sin(x)')
ax2.set_title('normalized')
ax2.set_xlabel('X')
ax2.set_ylabel('Amplitude')
ax2.grid(True)
ax2.legend()
plt.show()
'''
