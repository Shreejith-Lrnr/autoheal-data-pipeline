#!/usr/bin/env python3
"""
Test script for API data healing functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import ProductionDatabase
from agents import DataQualityAgent, DataHealingAgent
import pandas as pd
import requests
from datetime import datetime

def test_api_healing():
    """Test the complete API data healing pipeline"""

    # Initialize components
    db = ProductionDatabase()
    quality_agent = DataQualityAgent()
    healing_agent = DataHealingAgent()

    print("🧪 Testing API Data Healing Pipeline")
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

    # Step 3: Clean DataFrame for JSON serialization
    print("\n3. Cleaning DataFrame for storage...")
    cleaned_df = api_df.copy()
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            sample_values = cleaned_df[col].dropna().head(3)
            if any(isinstance(val, (dict, list)) for val in sample_values):
                cleaned_df[col] = cleaned_df[col].astype(str)
    print("✅ DataFrame cleaned for JSON serialization")

    # Step 4: Store dataset
    print("\n4. Storing dataset...")
    try:
        dataset_id = db.store_dataset(
            f"API_{api_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            cleaned_df
        )
        print(f"✅ Dataset stored with ID: {dataset_id}")
    except Exception as e:
        print(f"❌ Failed to store dataset: {e}")
        return

    # Step 5: Quality Analysis
    print("\n5. Analyzing data quality...")
    try:
        quality_report = quality_agent.analyze_dataset(api_df, f"API_{api_name}")
        print(f"✅ Quality analysis complete. Score: {quality_report.get('quality_score', 'N/A')}")
        print(f"   Issues found: {len(quality_report.get('issues', []))}")
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        return

    # Step 6: Generate Healing Plan
    print("\n6. Generating healing plan...")
    try:
        healing_plan = healing_agent.propose_healing_actions(api_df, quality_report)
        print(f"✅ Healing plan generated with {len(healing_plan.get('actions', []))} actions")
    except Exception as e:
        print(f"❌ Healing plan generation failed: {e}")
        return

    # Step 7: Execute Healing Actions
    print("\n7. Executing healing actions...")
    try:
        healed_df = api_df.copy()
        actions_executed = 0
        for action in healing_plan.get('actions', []):
            method = action.get('recommended_method', action.get('method', 'DELETE_ROWS'))
            healed_df, exec_log = healing_agent.execute_healing_action(healed_df, action, method)
            actions_executed += 1
        print(f"✅ {actions_executed} healing actions executed successfully")
    except Exception as e:
        print(f"❌ Healing execution failed: {e}")
        return

    # Step 8: Validation
    print("\n8. Validating healed data...")
    try:
        final_quality_report = quality_agent.analyze_dataset(healed_df, f"HEALED_API_{api_name}")

        validation_report = {
            'original_quality_score': quality_report.get('quality_score', 0),
            'final_quality_score': final_quality_report.get('quality_score', 0),
            'quality_improvement': final_quality_report.get('quality_score', 0) - quality_report.get('quality_score', 0),
            'original_issues_count': len(quality_report.get('issues', [])),
            'final_issues_count': len(final_quality_report.get('issues', [])),
            'issues_resolved': len(quality_report.get('issues', [])) - len(final_quality_report.get('issues', [])),
            'validation_timestamp': datetime.now().isoformat(),
            'healing_actions_executed': len(healing_plan.get('actions', [])),
            'validation_status': 'PASSED' if final_quality_report.get('quality_score', 0) >= quality_report.get('quality_score', 0) else 'IMPROVEMENT_NEEDED'
        }

        print("✅ Validation complete:")
        print(f"   Original quality: {validation_report['original_quality_score']:.2f}")
        print(f"   Final quality: {validation_report['final_quality_score']:.2f}")
        print(f"   Improvement: {validation_report['quality_improvement']:.2f}")
        print(f"   Issues resolved: {validation_report['issues_resolved']}")
        print(f"   Status: {validation_report['validation_status']}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return

    # Step 9: Store healed result
    print("\n9. Storing healed result...")
    try:
        healed_dataset_id = db.store_dataset(
            f"HEALED_API_{api_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            healed_df,
            None,
            validation_report.get('final_quality_score', 0)
        )
        print(f"✅ Healed data stored with ID: {healed_dataset_id}")
    except Exception as e:
        print(f"❌ Failed to store healed data: {e}")
        return

    print("\n🎉 API Data Healing Pipeline Test COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Original records: {len(api_df)}")
    print(f"Healed records: {len(healed_df)}")
    print(f"Quality improvement: {validation_report['quality_improvement']:.2f} points")
    print(f"Issues resolved: {validation_report['issues_resolved']}")

if __name__ == "__main__":
    test_api_healing()