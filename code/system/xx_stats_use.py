import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi
import os

# Database credentials
log_org = "know"
log_token = "Xp_sTOcg-46FFiQuplxz-Fqi-jEe5YGfOZarPR7gwZ4CMTMYseUPUjdKtp2xKV9w85TlBlh5X_lnaNzKULAhog=="
log_bucket = "log_bucket"
log_host = "https://eu-central-1-1.aws.cloud2.influxdata.com"

# Create a client instance
client = InfluxDBClient(url=log_host, token=log_token, org=log_org)
query_api = client.query_api()

# Define all mesa_stat variables
mesa_stat_fields = [
    "mesa_stat_create_atlas", "mesa_stat_edit_atlas", "mesa_stat_import_assets",
    "mesa_stat_import_atlas", "mesa_stat_import_geocodes", "mesa_stat_import_lines",
    "mesa_stat_process", "mesa_stat_process_lines", "mesa_stat_setup", "mesa_stat_startup"
]

# Include mesa_version in the query fields
query_parts = [f'r._field == "{field}"' for field in mesa_stat_fields] + ['r._field == "mesa_version"']
combined_query_condition = " or ".join(query_parts)
query_tbl_usage = f"""
from(bucket: "{log_bucket}")
  |> range(start: -180d)
  |> filter(fn: (r) => r._measurement == "tbl_usage")
  |> filter(fn: (r) => {combined_query_condition})
"""

# Fetch data and convert to pandas DataFrame
def fetch_data(query):
    result = query_api.query(org=log_org, query=query)
    records = []
    for table in result:
        for record in table.records:
            record_dict = {
                "UUID": record.values.get("uuid"),
                record.get_field(): record.get_value(),
                "Time": pd.to_datetime(record.get_time()).tz_localize(None)  # Convert to naive datetime here
            }
            records.append(record_dict)
    df = pd.DataFrame(records)

    # Convert all mesa_stat_ fields to numeric
    for field in mesa_stat_fields:
        df[field] = pd.to_numeric(df.get(field, pd.Series()), errors='coerce')

    # Group by UUID and sum all mesa_stat_ fields, also get min and max time, and aggregate mesa_version
    grouped_df = df.groupby('UUID').agg({
        **{field: 'sum' for field in mesa_stat_fields},
        "Time": ['min', 'max'],
        "mesa_version": lambda x: list(set(x.dropna()))
    }).reset_index()

    # Flatten the MultiIndex in columns
    grouped_df.columns = ['UUID', *mesa_stat_fields, 'Time_Min', 'Time_Max', 'Mesa_Version']

    # Calculate time difference in hours
    grouped_df['Time_Difference_Hours'] = (grouped_df['Time_Max'] - grouped_df['Time_Min']).dt.total_seconds() // 3600

    # Convert sums to integers and Time_Difference_Hours
    grouped_df[mesa_stat_fields] = grouped_df[mesa_stat_fields].fillna(0).astype(int)
    grouped_df['Time_Difference_Hours'] = grouped_df['Time_Difference_Hours'].astype(int)

    # Ensure the Mesa_Version column is placed right after UUID
    column_order = ['UUID', 'Mesa_Version'] + mesa_stat_fields + ['Time_Min', 'Time_Max', 'Time_Difference_Hours']
    grouped_df = grouped_df[column_order]

    return grouped_df

# Display data for tbl_usage
df_usage = fetch_data(query_tbl_usage)

# Ensure output folder exists
output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save to Excel
output_path = os.path.join(output_dir, "usagestats.xlsx")
df_usage.to_excel(output_path, sheet_name='tbl_usage', index=False)

# Close the client
client.close()

print(f"Data exported successfully to {output_path}")
