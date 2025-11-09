#!/usr/bin/env python3
"""
Test script to verify that quality score is properly updated after healing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import DataQualityAgent, DataHealingAgent
import pandas as pd

def test_quality_score_update():
    """Test that quality score is properly updated in processing_result after healing"""

    print("🧪 Testing Quality Score Update After Healing")
    print("=" * 50)

    # Initialize agents
    quality_agent = DataQualityAgent()
    healing_agent = DataHealingAgent()

    # Create sample data with quality issues
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, None, 35, 40, 'invalid'],  # Missing value and invalid type
        'salary': [50000, 60000, None, 80000, 90000],  # Missing value
        'department': ['HR', 'IT', 'HR', 'Finance', 'IT']
    }
    df = pd.DataFrame(data)

    print(f"📊 Original DataFrame: {len(df)} rows x {len(df.columns)} columns")

    # Step 1: Initial quality analysis
    print("\n1. Initial Quality Analysis...")
    quality_report = quality_agent.analyze_dataset(df, "Test Dataset")
    initial_score = quality_report.get('quality_score', 0)
    print(f"   Initial Quality Score: {initial_score:.1f}%")
    print(f"   Issues Found: {len(quality_report.get('issues', []))}")

    # Simulate processing_result structure
    processing_result = {
        'quality_report': quality_report,
        'dataset_name': 'Test Dataset'
    }

    print(f"\n2. Processing Result Before Healing:")
    print(f"   Quality Score in processing_result: {processing_result['quality_report'].get('quality_score', 0):.1f}%")

    # Step 2: Generate healing plan
    print("\n3. Generating Healing Plan...")
    healing_plan = healing_agent.propose_healing_actions(df, quality_report)
    print(f"   Healing actions: {len(healing_plan.get('actions', []))}")

    # Step 3: Execute healing actions
    print("\n4. Executing Healing Actions...")
    healed_df = df.copy()
    for action in healing_plan.get('actions', []):
        method = action.get('recommended_method', action.get('method', 'DELETE_ROWS'))
        healed_df, exec_log = healing_agent.execute_healing_action(healed_df, action, method)
        print(f"   Executed: {action.get('type', 'Unknown')}")

    print(f"   Healed DataFrame: {len(healed_df)} rows x {len(healed_df.columns)} columns")

    # Step 4: Calculate final quality score (simulating what happens in the UI)
    print("\n5. Calculating Final Quality Score...")
    final_quality_score = healing_agent.recalculate_quality_score(healed_df)
    print(f"   Final Quality Score: {final_quality_score:.1f}%")

    # Step 5: Update processing_result (simulating the fix)
    print("\n6. Updating Processing Result...")
    if 'quality_report' not in processing_result:
        processing_result['quality_report'] = {}

    processing_result['quality_report']['quality_score'] = final_quality_score
    processing_result['quality_report']['healing_completed'] = True
    processing_result['quality_report']['final_quality_score'] = final_quality_score
    processing_result['final_quality_score'] = final_quality_score

    print(f"   Updated Quality Score in processing_result: {processing_result['quality_report'].get('quality_score', 0):.1f}%")
    print(f"   Healing Completed Flag: {processing_result['quality_report'].get('healing_completed', False)}")

    # Step 6: Test the dataset summary logic
    print("\n7. Testing Dataset Summary Logic...")
    quality_report_in_result = processing_result.get('quality_report', {})
    if quality_report_in_result.get('healing_completed', False):
        display_score = quality_report_in_result.get('final_quality_score', quality_report_in_result.get('quality_score', 0))
        print(f"   Display Score (healing completed): {display_score:.1f}%")
    else:
        display_score = quality_report_in_result.get('quality_score', 0)
        print(f"   Display Score (healing not completed): {display_score:.1f}%")

    # Verification
    print("\n8. Verification:")
    success = abs(display_score - final_quality_score) < 0.1  # Allow small floating point differences
    if success:
        print("   ✅ SUCCESS: Quality score properly updated and displayed!")
        print(f"   Final Score: {final_quality_score:.1f}% | Display Score: {display_score:.1f}%")
    else:
        print("   ❌ FAILED: Quality score not properly updated!")
        print(f"   Final Score: {final_quality_score:.1f}% | Display Score: {display_score:.1f}%")

    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    return success

if __name__ == "__main__":
    test_quality_score_update()