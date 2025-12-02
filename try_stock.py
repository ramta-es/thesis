from alpha_vantage.timeseries import TimeSeries
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pywt

API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
ticker = 'AAPL'
num_days = 1500  # Set the number of days you want (None for all available data)
max_retries = 3
retry_delay = 60  # seconds

for attempt in range(max_retries):
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        if not data.empty:
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })

            # Limit to specified number of days
            if num_days is not None:
                data = data.head(num_days)

            # Convert OHLC to logarithmic values
            data['Open'] = np.log(data['Open'])
            data['High'] = np.log(data['High'])
            data['Low'] = np.log(data['Low'])
            data['Close'] = np.log(data['Close'])

            mpf.plot(data, type='candle', style='charles', title=f'{ticker} Candlestick Chart (Log Scale)', volume=True)

            # Create grayscale image
            ohlc_data = data[['Open', 'High', 'Low', 'Close']].values
            # Normalize to 0-65535 range for 16-bit grayscale
            min_val = ohlc_data.min()
            max_val = ohlc_data.max()
            normalized = ((ohlc_data - min_val) / (max_val - min_val) * (2 ** 16 - 1)).astype(np.uint16)

            # Create and display image (4 rows x number of days columns)
            plt.figure(figsize=(12, 4))
            plt.imshow(normalized.T, cmap='gray', aspect='auto', interpolation='nearest', vmin=0, vmax=65535)
            plt.colorbar(label='Intensity (0-65535)')
            plt.xlabel('Time (Days)')
            plt.ylabel('OHLC Values')
            plt.yticks([0, 1, 2, 3], ['Open', 'High', 'Low', 'Close'])
            plt.title(f'{ticker} OHLC Grayscale Image')
            plt.tight_layout()
            plt.show()

            break
        else:
            print(f"Attempt {attempt + 1}: No data. Retrying in {retry_delay} seconds...")
    except Exception as e:
        print(f"Attempt {attempt + 1}: Error: {e}. Retrying in {retry_delay} seconds...")
    time.sleep(retry_delay)
else:
    print("Failed to download data after several attempts. Try again later.")
######################################


# Apply wavelet transform to each OHLC column
wavelet = 'db4'  # Daubechies 4 wavelet
level = 5  # Decomposition level

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
columns = ['Open', 'High', 'Low', 'Close']

for idx, col in enumerate(columns):
    # Get the data for this column
    signal = data[col].values

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Reconstruct and plot
    axes[idx].plot(signal, label='Original', alpha=0.7)

    # Plot approximation (low frequency component)
    reconstructed = pywt.waverec(coeffs, wavelet)[:len(signal)]
    axes[idx].plot(reconstructed, label='Wavelet Reconstruction', alpha=0.7)

    axes[idx].set_title(f'{col} - Wavelet Transform ({wavelet})')
    axes[idx].set_xlabel('Time (Days)')
    axes[idx].set_ylabel('Log Value')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display wavelet coefficients as images
fig, axes = plt.subplots(4, level + 1, figsize=(15, 10))

for idx, col in enumerate(columns):
    signal = data[col].values
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    for j, coeff in enumerate(coeffs):
        # Normalize coefficients for visualization
        coeff_norm = np.abs(coeff)
        coeff_norm = ((coeff_norm - coeff_norm.min()) / (coeff_norm.max() - coeff_norm.min() + 1e-10) * 255).astype(
            np.uint8)

        axes[idx, j].imshow(coeff_norm.reshape(1, -1), cmap='gray', aspect='auto', interpolation='nearest')
        axes[idx, j].set_title(f'Level {j}' if j > 0 else 'Approx')
        axes[idx, j].set_yticks([])
        axes[idx, j].set_xlabel('Coefficients')

    axes[idx, 0].set_ylabel(col)

plt.suptitle(f'{ticker} Wavelet Coefficients')
plt.tight_layout()
plt.show()

# Apply FFT to each OHLC column
fig, axes = plt.subplots(4, 2, figsize=(15, 10))
columns = ['Open', 'High', 'Low', 'Close']

for idx, col in enumerate(columns):
    signal = data[col].values

    # Compute FFT
    fft_values = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal))

    # Get magnitude and phase
    magnitude = np.abs(fft_values)
    phase = np.angle(fft_values)

    # Plot original signal
    axes[idx, 0].plot(signal)
    axes[idx, 0].set_title(f'{col} - Time Domain')
    axes[idx, 0].set_xlabel('Time (Days)')
    axes[idx, 0].set_ylabel('Value')
    axes[idx, 0].grid(True, alpha=0.3)

    # Plot frequency spectrum (magnitude)
    axes[idx, 1].plot(fft_freq[:len(fft_freq) // 2], magnitude[:len(magnitude) // 2])
    axes[idx, 1].set_title(f'{col} - Frequency Domain (FFT Magnitude)')
    axes[idx, 1].set_xlabel('Frequency')
    axes[idx, 1].set_ylabel('Magnitude')
    axes[idx, 1].grid(True, alpha=0.3)

plt.suptitle(f'{ticker} FFT Analysis')
plt.tight_layout()
plt.show()

# Display FFT coefficients as grayscale image
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for idx, col in enumerate(columns):
    signal = data[col].values
    fft_values = np.fft.fft(signal)
    magnitude = np.abs(fft_values)

    # Normalize magnitude for visualization (0-65535 range)
    mag_norm = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10) * 65535).astype(np.uint16)

    row = idx // 2
    col_idx = idx % 2

    axes[row, col_idx].imshow(mag_norm.reshape(1, -1), cmap='gray', aspect='auto', interpolation='nearest', vmin=0,
                              vmax=65535)
    axes[row, col_idx].set_title(f'{col} FFT Magnitude')
    axes[row, col_idx].set_xlabel('Frequency Bins')
    axes[row, col_idx].set_yticks([])
    axes[row, col_idx].colorbar = plt.colorbar(axes[row, col_idx].images[0], ax=axes[row, col_idx],
                                               label='Intensity (0-65535)')

plt.suptitle(f'{ticker} FFT Coefficients (16-bit Grayscale)')
plt.tight_layout()
plt.show()
