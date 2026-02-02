import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def analyze_power_data(filepath):
    """
    Analyzes power meter data from a tab-delimited CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
    """
    print(f"Loading data from {filepath}...")
    
    try:
        # Step 1: Detect header row
        header_row_idx = None
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.strip().startswith("Samples"):
                    header_row_idx = i
                    break
        
        if header_row_idx is None:
            print("Error: Could not find header row starting with 'Samples'.")
            # Fallback to user specified 15th row (index 14) if search fails, though likely won't work
            header_row_idx = 14
        else:
            print(f"Found header at row {header_row_idx + 1} (index {header_row_idx})")

        # Step 2: Load data
        # Use skiprows to accurately target the physical line number found
        # header=0 is default, meaning the first line after skipping is header
        df = pd.read_csv(filepath, sep='\t', skiprows=header_row_idx)
        
        # Step 3: Clean column names
        # Remove leading/trailing whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        
        # Verify columns
        if len(df.columns) < 5:
            print(f"Error: DataFrame has {len(df.columns)} columns, expected at least 5. Columns: {df.columns}")
            return

        # Extract data columns by index (0-based)
        # 3rd column: Time (index 2)
        # 4th column: Power Meter 1 (index 3)
        # 5th column: Power Meter 2 (index 4)
        
        # Note: If there are empty columns at the end (due to trailing tabs), indices might be shifted?
        # Let's rely on indices as requested.
        
        time_col_name = df.columns[2]
        p1_col_name = df.columns[3]
        p2_col_name = df.columns[4]
        
        print(f"Using columns by index: Time='{time_col_name}', P1='{p1_col_name}', P2='{p2_col_name}'")

        # Step 4: Clean data (strip whitespace from strings)
        # Convert object columns to string and strip
        for col in [time_col_name]:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

        # Step 5: Parse Time
        # The file has a 'Date (MM/dd/yyyy)' column at index 1 usually. 
        # Let's try to use it if available to support multi-day data.
        date_col_name = df.columns[1]
        
        try:
            # Check if we can parse combined Date + Time
            # Assuming Date is col 1 and Time is col 2
            combined_datetime_str = df[date_col_name].astype(str).str.strip() + " " + df[time_col_name]
            df['Parsed_Time'] = pd.to_datetime(combined_datetime_str, format='mixed', errors='coerce')
            
            # If that failed for all (all NaT), try just Time
            if df['Parsed_Time'].isna().all():
                 print("Date+Time parsing failed, falling back to Time only.")
                 df['Parsed_Time'] = pd.to_datetime(df[time_col_name], format='mixed', errors='coerce')
                 
        except Exception as e:
            print(f"Date+Time parsing exception: {e}. Falling back to time column only.")
            try:
                df['Parsed_Time'] = pd.to_datetime(df[time_col_name], format='mixed')
            except Exception as e2:
                 print(f"Error parsing time column: {e2}")
                 return

        # Step 6: Ensure numeric data for Power
        # Handle scientific notation and potential whitespace
        df[p1_col_name] = pd.to_numeric(df[p1_col_name], errors='coerce')
        df[p2_col_name] = pd.to_numeric(df[p2_col_name], errors='coerce')
        
        # Drop rows with NaNs in critical columns
        original_len = len(df)
        df.dropna(subset=['Parsed_Time', p1_col_name, p2_col_name], inplace=True)
        if len(df) < original_len:
            print(f"Dropped {original_len - len(df)} rows due to invalid/missing data.")
        
        if len(df) == 0:
            print("No valid data rows remaining.")
            return

        # Calculations
        # Metric: sqrt( (P2 / P1) * (1 / 5.2) ) * 100%
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (df[p2_col_name] / df[p1_col_name]) / 5.2
            # Handle negative ratios
            ratio[ratio < 0] = 0
            df['Metric_Percent'] = np.sqrt(ratio) * 100.0

        # Plotting
        plt.style.use('seaborn-v0_8-darkgrid') 
        
        # Plot 1: Two traces vs Time
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df['Parsed_Time'], df[p1_col_name], label='Power Meter 1', color='tab:blue', linewidth=1.5)
        ax1.plot(df['Parsed_Time'], df[p2_col_name], label='Power Meter 2', color='tab:orange', linewidth=1.5)
        
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Power (W)')
        ax1.set_title(f'Power Traces vs Time\n{os.path.basename(filepath)}')
        ax1.legend()
        plt.tight_layout()
        plot1_path = os.path.join(os.path.dirname(filepath), 'plot_traces.png')
        plt.savefig(plot1_path)
        print(f"Saved plot 1 to {plot1_path}")
        plt.close(fig1)

        # Plot 2: Metric vs Time
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df['Parsed_Time'], df['Metric_Percent'], label='Calculated Metric', color='tab:green', linewidth=1.5)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Metric (%)')
        ax2.set_title(rf'Metric: $\sqrt{{P2 / (P1 \cdot 5.2)}} \times 100\%$\n{os.path.basename(filepath)}')
        plt.tight_layout()
        plot2_path = os.path.join(os.path.dirname(filepath), 'plot_metric.png')
        plt.savefig(plot2_path)
        print(f"Saved plot 2 to {plot2_path}")
        plt.close(fig2)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        print("Usage: python analyze_power_data.py <path_to_csv>")
        target_file = "dummy_data.csv"
        
    if os.path.exists(target_file):
        analyze_power_data(target_file)
    else:
        print(f"File not found: {target_file}")
