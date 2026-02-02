import csv
import random
import datetime

filename = "dummy_data.csv"

# Metadata rows (lines 1-14)
timestamp = datetime.datetime.now()
header_lines = [
    ["# Dummy Data Generator"],
    ["# Created on", str(timestamp)],
    ["# Device: Test Power Meter"],
    ["#"],
    ["# Parameters:"],
    ["# P1_offset", "0.5"],
    ["# P2_offset", "0.2"],
    ["# Noise_level", "0.01"],
    ["#"],
    ["#"],
    ["#"],
    ["#"],
    ["#"],
    ["# End of metadata"]
]

# Column headers (line 15)
# Columns: Index, Date, Time, Power1, Power2, Status
col_headers = ["Index", "Date", "Time", "Power1 (W)", "Power2 (W)", "Status"]

# Generate data
data_rows = []
start_time = timestamp
for i in range(100):
    current_time = start_time + datetime.timedelta(seconds=i * 0.1)
    
    # Random power values ensuring P2/P1/5.2 is valid for sqrt (positive)
    # P1 around 1.0, P2 around 0.5
    p1 = 1.0 + random.uniform(-0.1, 0.1)
    p2 = 0.5 + random.uniform(-0.05, 0.05)
    
    time_str = current_time.strftime("%H:%M:%S.%f")[:-3] # HH:MM:SS.mmm
    date_str = current_time.strftime("%Y-%m-%d")
    
    row = [
        str(i),
        date_str,
        time_str,
        f"{p1:.6f}",
        f"{p2:.6f}",
        "OK"
    ]
    data_rows.append(row)

# Write to file
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    
    # Write metadata
    for line in header_lines:
        writer.writerow(line)
        
    # Write column headers (Row 15)
    writer.writerow(col_headers)
    
    # Write data (Row 16+)
    writer.writerows(data_rows)

print(f"Generated {filename} with {len(data_rows)} rows of data.")
