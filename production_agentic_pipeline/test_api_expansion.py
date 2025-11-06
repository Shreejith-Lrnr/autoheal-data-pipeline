import pandas as pd
import json

# Simulate Open-Meteo API response
payload = {
    'latitude': 52.52,
    'longitude': 13.41,
    'hourly': {
        'time': ['2025-10-20T00:00', '2025-10-20T01:00', '2025-10-20T02:00'],
        'temperature_2m': [15.2, 14.8, 14.5]
    }
}

print("Testing nested JSON expansion logic...")
print(f"Original payload keys: {list(payload.keys())}")
print(f"Hourly data: {payload['hourly']}")

# Test the logic from database.py
df = None
time_series_keys = ['hourly', 'daily', 'data', 'results', 'records', 'items']

for key in time_series_keys:
    if key in payload and isinstance(payload[key], dict):
        nested = payload[key]
        # Check if all values in nested dict are lists of same length
        if all(isinstance(v, list) for v in nested.values()):
            lengths = [len(v) for v in nested.values()]
            if len(set(lengths)) == 1 and lengths[0] > 1:
                # Expand nested time-series into rows
                df = pd.DataFrame(nested)
                # Add metadata columns from top level (if any)
                for meta_key, meta_val in payload.items():
                    if meta_key != key and not isinstance(meta_val, (dict, list)):
                        df[meta_key] = meta_val
                print(f"\n✅ Expanded '{key}' into {len(df)} rows")
                break

if df is not None:
    print("\nResulting DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
else:
    print("\n❌ Failed to expand data")
