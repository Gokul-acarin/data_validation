#!/usr/bin/env python3
"""
Show and analyze data quality anomalies.

This script compares original data files with anomaly-injected files to show
the differences and provide analysis of data quality issues.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def analyze_file_comparison(original_file: Path, anomaly_file: Path) -> Dict:
    """Analyze differences between original and anomaly files."""
    print(f"Analyzing: {original_file.name} vs {anomaly_file.name}")
    
    # Load data
    df_original = pd.read_csv(original_file)
    df_anomaly = pd.read_csv(anomaly_file)
    
    analysis = {
        'original_records': len(df_original),
        'anomaly_records': len(df_anomaly),
        'record_difference': len(df_anomaly) - len(df_original),
        'columns': list(df_original.columns),
        'null_analysis': {},
        'data_quality_issues': []
    }
    
    # Analyze null values for each column
    for column in df_original.columns:
        if column in df_anomaly.columns:
            original_nulls = df_original[column].isnull().sum()
            anomaly_nulls = df_anomaly[column].isnull().sum()
            null_increase = anomaly_nulls - original_nulls
            
            analysis['null_analysis'][column] = {
                'original_nulls': original_nulls,
                'anomaly_nulls': anomaly_nulls,
                'null_increase': null_increase,
                'original_null_pct': (original_nulls / len(df_original)) * 100,
                'anomaly_null_pct': (anomaly_nulls / len(df_anomaly)) * 100
            }
            
            if null_increase > 0:
                analysis['data_quality_issues'].append(
                    f"Missing values in {column}: +{null_increase} nulls"
                )
    
    # Detect invalid emails
    for column in df_original.columns:
        if 'email' in column.lower() and column in df_anomaly.columns:
            invalid_emails = detect_invalid_emails(df_anomaly[column])  # type: ignore
            if invalid_emails > 0:
                analysis['data_quality_issues'].append(
                    f"Invalid emails in {column}: {invalid_emails} invalid addresses"
                )
    
    # Detect invalid phone numbers
    for column in df_original.columns:
        if any(phone_keyword in column.lower() for phone_keyword in ['phone', 'tel', 'mobile']):
            if column in df_anomaly.columns:
                invalid_phones = detect_invalid_phones(df_anomaly[column])  # type: ignore
                if invalid_phones > 0:
                    analysis['data_quality_issues'].append(
                        f"Invalid phones in {column}: {invalid_phones} invalid numbers"
                    )
    
    return analysis


def detect_invalid_emails(series: pd.Series) -> int:
    """Detect invalid email addresses in a series."""
    import re
    
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    invalid_count = 0
    
    for value in series.dropna():
        if not email_pattern.match(str(value)):
            invalid_count += 1
    
    return invalid_count


def detect_invalid_phones(series: pd.Series) -> int:
    """Detect invalid phone numbers in a series."""
    import re
    
    # Basic phone pattern (adjust as needed)
    phone_pattern = re.compile(r'^[\+]?[1-9][\d]{0,15}$')
    invalid_count = 0
    
    for value in series.dropna():
        # Remove common separators
        cleaned = re.sub(r'[\s\-\(\)\.]', '', str(value))
        if not phone_pattern.match(cleaned):
            invalid_count += 1
    
    return invalid_count


def print_analysis_summary(analysis: Dict) -> None:
    """Print a summary of the analysis."""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Original records: {analysis['original_records']}")
    print(f"Anomaly records: {analysis['anomaly_records']}")
    print(f"Record difference: {analysis['record_difference']}")
    print(f"Columns analyzed: {len(analysis['columns'])}")
    
    print("\nNULL VALUE ANALYSIS:")
    print("-" * 40)
    for column, null_data in analysis['null_analysis'].items():
        if null_data['null_increase'] > 0:
            print(f"{column}:")
            print(f"  Original nulls: {null_data['original_nulls']} ({null_data['original_null_pct']:.1f}%)")
            print(f"  Anomaly nulls: {null_data['anomaly_nulls']} ({null_data['anomaly_null_pct']:.1f}%)")
            print(f"  Increase: +{null_data['null_increase']} nulls")
    
    if analysis['data_quality_issues']:
        print("\nDATA QUALITY ISSUES DETECTED:")
        print("-" * 40)
        for issue in analysis['data_quality_issues']:
            print(f"• {issue}")
    else:
        print("\nNo significant data quality issues detected.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Show and analyze data quality anomalies")
    parser.add_argument("original_files", nargs="+", help="Original CSV files")
    parser.add_argument("anomaly_files", nargs="+", help="Anomaly-injected CSV files")
    parser.add_argument("--output", "-o", type=Path, help="Output file for detailed analysis")
    
    args = parser.parse_args()
    
    if len(args.original_files) != len(args.anomaly_files):
        print("❌ Error: Number of original files must match number of anomaly files")
        sys.exit(1)
    
    print(f"Analyzing {len(args.original_files)} file pair(s)...")
    
    all_analyses = []
    
    # Analyze each pair of files
    for original_file, anomaly_file in zip(args.original_files, args.anomaly_files):
        original_path = Path(original_file)
        anomaly_path = Path(anomaly_file)
        
        if not original_path.exists():
            print(f"❌ Error: Original file not found: {original_file}")
            continue
        
        if not anomaly_path.exists():
            print(f"❌ Error: Anomaly file not found: {anomaly_file}")
            continue
        
        try:
            analysis = analyze_file_comparison(original_path, anomaly_path)
            all_analyses.append(analysis)
            print_analysis_summary(analysis)
        except Exception as e:
            print(f"❌ Error analyzing {original_file} vs {anomaly_file}: {e}")
    
    # Save detailed analysis if requested
    if args.output and all_analyses:
        try:
            import json
            with open(args.output, 'w') as f:
                json.dump(all_analyses, f, indent=2)
            print(f"\n✅ Detailed analysis saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
    
    print(f"\n✅ Analysis completed for {len(all_analyses)} file pair(s)")


if __name__ == "__main__":
    main() 