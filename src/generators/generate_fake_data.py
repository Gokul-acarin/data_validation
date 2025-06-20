#!/usr/bin/env python3
"""
Generate fake data using Faker library.

This script generates realistic test data based on field mappings from OpenAI.
"""

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from faker import Faker


def load_field_mappings(mappings_file: Path) -> List[Dict[str, Any]]:
    """Load field mappings from CSV file."""
    mappings = []
    with open(mappings_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse parameters if present
            parameters = None
            if row.get('parameters') and row['parameters'].strip():
                try:
                    parameters = json.loads(row['parameters'])
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse parameters for field {row['field_name']}")
            
            mappings.append({
                'field_name': row['field_name'],
                'faker_provider': row['faker_provider'],
                'parameters': parameters,
                'validation_rules': row['validation_rules']
            })
    
    if not mappings:
        raise ValueError("No field mappings found in file")
    
    print(f"Loaded {len(mappings)} field mappings")
    return mappings


def generate_field_value(faker: Faker, mapping: Dict[str, Any]) -> Any:
    """Generate a value for a specific field using Faker."""
    provider_name = mapping['faker_provider']
    parameters = mapping.get('parameters') or {}
    
    # Get the provider method from Faker
    if hasattr(faker, provider_name):
        provider_method = getattr(faker, provider_name)
        
        # Call the method with parameters if provided
        if parameters:
            if isinstance(parameters, dict):
                value = provider_method(**parameters)
            else:
                value = provider_method(parameters)
        else:
            value = provider_method()
        
        return value
    else:
        print(f"Warning: Unknown Faker provider '{provider_name}' for field '{mapping['field_name']}'")
        return f"Unknown provider: {provider_name}"


def generate_test_data(mappings_file: Path, num_records: int, output_file: Path, seed: Union[int, None] = None) -> None:
    """Generate realistic test data based on field mappings."""
    print(f"Generating {num_records} records...")
    print(f"Mappings file: {mappings_file}")
    print(f"Output file: {output_file}")
    
    # Load field mappings
    mappings = load_field_mappings(mappings_file)
    
    # Create Faker instance
    faker = Faker('en_US')
    if seed:
        Faker.seed(seed)
        random.seed(seed)
        print(f"Using seed: {seed}")
    
    # Generate data
    data = []
    for i in range(num_records):
        record = {}
        for mapping in mappings:
            field_name = mapping['field_name']
            value = generate_field_value(faker, mapping)
            record[field_name] = value
        
        data.append(record)
        
        # Show progress for large datasets
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} records...")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Successfully generated {num_records} records")
    print(f"Output saved to: {output_file}")


def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Usage: python generate_fake_data.py <num_records> <mappings_file> <output_file>")
        print("Example: python generate_fake_data.py 100 output/mappings/field_mappings.csv data/generated/fake_data_100.csv")
        sys.exit(1)
    
    try:
        num_records = int(sys.argv[1])
        mappings_file = Path(sys.argv[2])
        output_file = Path(sys.argv[3])
        
        if not mappings_file.exists():
            print(f"❌ Error: Mappings file not found: {mappings_file}")
            sys.exit(1)
        
        generate_test_data(mappings_file, num_records, output_file)
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 