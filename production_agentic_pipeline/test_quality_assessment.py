#!/usr/bin/env python3
"""
Test script for API data quality assessment display
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import ProductionDatabase
from agents import DataQualityAgent
import pandas as pd
import requests

def test_quality_assessment():
    """Test the quality assessment logic for API data"""

    # Initialize components
    db = ProductionDatabase()
    quality_agent = DataQualityAgent()

    print("🧪 Testing API Data Quality Assessment")
    print("=" * 50)

    # Step 1: Add a test API source
    print("\n1. Adding test API source...")
    api_url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&hourly=temperature_2m,relative_humidity_2m,precipitation"
    api_name = "Test NYC Weather API"

    try:
        source_id = db.register_api_source(api_name, api_url, 300)  # 5 minutes
        print(f"✅ API source registered with ID: {source_id}")
    except Exception as e:
        print(f"❌ Failed to register API source: {e}")
        return

    # Step 2: Fetch API data
    print("\n2. Fetching API data...")
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API data fetched successfully: {len(data)} records")

            # Convert to DataFrame
            if 'hourly' in data:
                hourly_data = data['hourly']
                api_df = pd.DataFrame(hourly_data)
                print(f"✅ Data converted to DataFrame: {len(api_df)} rows x {len(api_df.columns)} columns")
            else:
                print("❌ Unexpected API response structure")
                return
        else:
            print(f"❌ API request failed: HTTP {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Failed to fetch API data: {e}")
        return

    # Step 3: Quality Analysis
    print("\n3. Analyzing data quality...")
    try:
        quality_report = quality_agent.analyze_dataset(api_df, f"API_{api_name}")
        quality_score = quality_report.get('quality_score', 0)
        issues_count = len(quality_report.get('issues', []))

        print(f"✅ Quality analysis complete!")
        print(f"   Quality Score: {quality_score:.1f}%")
        print(f"   Issues Found: {issues_count}")

        # Simulate the UI logic
        if quality_score >= 90:
            status = "Excellent"
            emoji = "🟢"
            button_text = "✨ Optimize This High-Quality API Data"
        elif quality_score >= 70:
            status = "Good"
            emoji = "🟡"
            button_text = "🩺 Improve This API Data Quality"
        else:
            status = "Needs Healing"
            emoji = "🔴"
            button_text = f"🩺 Auto-Heal This API Data (Quality: {quality_score:.1f}%)"

        print(f"   Status: {emoji} {status}")
        print(f"   Button Text: {button_text}")

        # Show top issues
        if issues_count > 0:
            print(f"\n   Top Issues:")
            issues_list = quality_report.get('issues', [])[:3]  # Show top 3
            for i, issue in enumerate(issues_list, 1):
                issue_type = issue.get('type', 'Unknown')
                severity = issue.get('severity', 'medium')
                description = issue.get('description', 'Data quality issue detected')

                severity_emoji = "🔴" if severity == 'high' else "🟡" if severity == 'medium' else "ℹ️"
                print(f"     {i}. {severity_emoji} {issue_type}: {description}")

    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        return

    print("\n🎉 Quality Assessment Test COMPLETED!")
    print("=" * 50)

if __name__ == "__main__":
    test_quality_assessment()