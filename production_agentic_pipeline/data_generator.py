import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_realistic_sales_data(num_records=10000):
    """Generate realistic sales data with intentional quality issues"""
    
    # Base data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 9, 15)
    
    data = []
    for i in range(num_records):
        # Introduce realistic issues
        order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        
        # NULL values (5% of data)
        quantity = None if random.random() < 0.05 else random.randint(1, 100)
        
        # Zero values (2% of data) 
        if quantity is not None and random.random() < 0.02:
            quantity = 0
            
        # Negative prices (1% of data)
        unit_price = round(random.uniform(10, 500), 2)
        if random.random() < 0.01:
            unit_price = -unit_price
            
        # Missing regions (3% of data)
        region = None if random.random() < 0.03 else random.choice(['North', 'South', 'East', 'West', 'Central'])
        
        data.append({
            'OrderID': f'ORD-{i+1:06d}',
            'CustomerID': f'CUST-{random.randint(1, 1000):04d}',
            'ProductID': f'PROD-{random.randint(1, 50):03d}',
            'OrderDate': order_date.strftime('%Y-%m-%d'),
            'Quantity': quantity,
            'UnitPrice': unit_price,
            'Region': region,
            'SalesRep': f'Rep-{random.randint(1, 20):02d}',
            'Customer': f'Company {chr(65 + i % 26)}'
        })
    
    return pd.DataFrame(data)

# Generate and save sample datasets
if __name__ == "__main__":
    # Sales data
    sales_df = generate_realistic_sales_data(10000)
    sales_df.to_csv('sample_data/sales_data.csv', index=False)
    
    print("✅ Generated realistic sample data with quality issues")
    print(f"Records: {len(sales_df)}")
    print(f"NULL values: {sales_df.isnull().sum().sum()}")
    print(f"Duplicates: {sales_df.duplicated().sum()}")
