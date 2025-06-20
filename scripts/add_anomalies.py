#!/usr/bin/env python3
"""
Add data quality anomalies to generated test data.

This script takes generated CSV files and injects various types of data quality
anomalies to test data validation rules.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

import pandas as pd


def add_missing_values(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """Add missing values (NULL) to the dataframe."""
    df_anomaly = df.copy()
    total_cells = len(df) * len(df.columns)
    num_missing = int(total_cells * percentage)
    
    # Randomly select cells to make missing
    rows = random.sample(range(len(df)), num_missing)
    cols = random.choices(list(df.columns), k=num_missing)
    
    for row, col in zip(rows, cols):
        df_anomaly.at[row, col] = None
    
    print(f"Added {num_missing} missing values ({percentage*100:.1f}% of total cells)")
    return df_anomaly


def add_invalid_emails(df: pd.DataFrame, column: str, percentage: float) -> pd.DataFrame:
    """Add invalid email addresses."""
    if column not in df.columns:
        print(f"Warning: Email column '{column}' not found")
        return df
    
    df_anomaly = df.copy()
    email_mask = df[column].notna() & df[column].str.contains('@', na=False)
    email_indices = df[email_mask].index.tolist()
    
    if not email_indices:
        print(f"Warning: No valid emails found in column '{column}'")
        return df
    
    num_invalid = int(len(email_indices) * percentage)
    invalid_indices = random.sample(email_indices, num_invalid)
    
    invalid_emails = [
        "invalid.email",
        "missing@domain",
        "@nodomain.com",
        "spaces @domain.com",
        "double@@domain.com",
        "domain@.com",
        ".domain@com",
        "domain@com.",
        "domain@com..",
        "domain@com@domain.com"
    ]
    
    for idx in invalid_indices:
        df_anomaly.at[idx, column] = random.choice(invalid_emails)
    
    print(f"Added {num_invalid} invalid emails to column '{column}'")
    return df_anomaly


def add_invalid_phone_numbers(df: pd.DataFrame, column: str, percentage: float) -> pd.DataFrame:
    """Add invalid phone numbers."""
    if column not in df.columns:
        print(f"Warning: Phone column '{column}' not found")
        return df
    
    df_anomaly = df.copy()
    phone_mask = df[column].notna()
    phone_indices = df[phone_mask].index.tolist()
    
    if not phone_indices:
        print(f"Warning: No phone numbers found in column '{column}'")
        return df
    
    num_invalid = int(len(phone_indices) * percentage)
    invalid_indices = random.sample(phone_indices, num_invalid)
    
    invalid_phones = [
        "123",
        "12345678901234567890",  # Too long
        "abc-def-ghij",
        "123-456-789a",
        "123.456.7890",
        "+1-123-456-7890-1234",  # Too many digits
        "123-456-789",  # Too few digits
        "123-456-7890-1234",  # Extra dash
        "1234567890",  # No formatting
        "123-456-7890-1234-5678"  # Way too long
    ]
    
    for idx in invalid_indices:
        df_anomaly.at[idx, column] = random.choice(invalid_phones)
    
    print(f"Added {num_invalid} invalid phone numbers to column '{column}'")
    return df_anomaly


def add_anomalies_to_file(data_file: Path, output_dir: Path, percentage: float) -> None:
    """Add anomalies to a single CSV file."""
    print(f"Processing file: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Apply anomalies
    df_anomaly = df.copy()
    
    # Missing values (apply to all columns)
    df_anomaly = add_missing_values(df_anomaly, percentage * 0.3)
    
    # Email-specific anomalies
    email_columns = [col for col in df.columns if 'email' in col.lower()]
    for col in email_columns:
        df_anomaly = add_invalid_emails(df_anomaly, col, percentage * 0.2)
    
    # Phone-specific anomalies
    phone_columns = [col for col in df.columns if any(phone_keyword in col.lower() 
                                                     for phone_keyword in ['phone', 'tel', 'mobile'])]
    for col in phone_columns:
        df_anomaly = add_invalid_phone_numbers(df_anomaly, col, percentage * 0.2)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_stem = data_file.stem
    output_filename = f"{input_stem}_with_anomalies_{int(percentage * 100)}pct.csv"
    output_file = output_dir / output_filename
    
    # Save the data with anomalies
    df_anomaly.to_csv(output_file, index=False)
    print(f"✅ Saved file with anomalies: {output_file}")
    print(f"Original records: {len(df)}, Anomaly records: {len(df_anomaly)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Add data quality anomalies to CSV files")
    parser.add_argument("data_files", nargs="+", help="CSV files to process")
    parser.add_argument("--percentage", "-p", type=float, default=0.15,
                       help="Percentage of data to inject with anomalies (default: 0.15)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("data/with_anomalies"),
                       help="Output directory for files with anomalies")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print(f"Adding anomalies to {len(args.data_files)} file(s)")
    print(f"Anomaly percentage: {args.percentage * 100:.1f}%")
    print(f"Output directory: {args.output_dir}")
    
    # Process each file
    for data_file in args.data_files:
        file_path = Path(data_file)
        if not file_path.exists():
            print(f"❌ Error: File not found: {data_file}")
            continue
        
        try:
            add_anomalies_to_file(file_path, args.output_dir, args.percentage)
        except Exception as e:
            print(f"❌ Error processing {data_file}: {e}")
    
    print("✅ Anomaly injection completed!")


if __name__ == "__main__":
    main() 