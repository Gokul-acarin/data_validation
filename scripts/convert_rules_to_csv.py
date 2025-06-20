#!/usr/bin/env python3
"""
Convert validation rules from text format to structured CSV format.

This script reads a validation rules text file and converts it to a CSV format
that can be used by the field mapping generator.
"""

import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple


def parse_validation_rules(rules_text: str) -> List[Tuple[str, str]]:
    """
    Parse validation rules from text format.
    
    Args:
        rules_text: Raw text containing validation rules
        
    Returns:
        List of tuples containing (field_name, field_description)
    """
    # Split by double newlines to separate field definitions
    sections = re.split(r'\n\s*\n', rules_text.strip())
    
    rules = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Split into lines
        lines = section.split('\n')
        if len(lines) < 2:
            print(f"Warning: Skipping malformed section: {section}")
            continue
        
        # First line should be the field name
        field_name = lines[0].strip().rstrip(':')
        
        # Remaining lines form the description
        description = '\n'.join(lines[1:]).strip()
        
        if field_name and description:
            rules.append((field_name, description))
            print(f"Parsed rule: {field_name}")
    
    if not rules:
        raise ValueError("No valid rules found in the text")
    
    print(f"Successfully parsed {len(rules)} validation rules")
    return rules


def convert_rules_to_csv(rules_file: Path, output_file: Path) -> None:
    """
    Convert validation rules from text file to CSV format.
    
    Args:
        rules_file: Path to the input rules text file
        output_file: Path to the output CSV file
    """
    print(f"Converting validation rules to CSV...")
    print(f"Input file: {rules_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Read the rules file
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_text = f.read()
        
        # Parse the rules
        rules = parse_validation_rules(rules_text)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['field_name', 'field_description'])
            writer.writerows(rules)
        
        print(f"✅ Successfully converted {len(rules)} rules to CSV")
        print(f"Output saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: Rules file not found: {rules_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error converting rules to CSV: {e}")
        sys.exit(1)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python convert_rules_to_csv.py <rules_file>")
        print("Example: python convert_rules_to_csv.py config/validation_rules.txt")
        sys.exit(1)
    
    rules_file = Path(sys.argv[1])
    output_file = Path("output/mappings/field_descriptions.csv")
    
    convert_rules_to_csv(rules_file, output_file)


if __name__ == "__main__":
    main() 