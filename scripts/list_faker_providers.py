#!/usr/bin/env python3
"""
List all available Faker providers.

This script displays all available Faker provider methods organized by category
for reference and debugging purposes.
"""

import sys
from typing import Dict, List

from faker import Faker


def get_all_faker_providers() -> Dict[str, List[str]]:
    """Get all available Faker provider methods organized by category."""
    faker = Faker('en_US')
    
    # Define provider categories
    categories = {
        "Names & Personal Info": [
            'name', 'first_name', 'last_name', 'full_name', 'name_male', 'name_female',
            'prefix', 'prefix_male', 'prefix_female', 'suffix', 'suffix_male', 'suffix_female'
        ],
        "Address & Location": [
            'address', 'street_address', 'street_name', 'street_suffix', 'city', 'state',
            'state_abbr', 'country', 'country_code', 'postcode', 'zipcode', 'latitude', 'longitude'
        ],
        "Contact Information": [
            'email', 'safe_email', 'free_email', 'company_email', 'phone_number', 'cellphone',
            'country_calling_code'
        ],
        "Dates & Time": [
            'date_of_birth', 'date', 'time', 'year', 'month', 'day_of_week', 'day_of_month',
            'timezone'
        ],
        "Company & Business": [
            'company', 'company_suffix', 'job', 'job_title', 'company_name', 'catch_phrase',
            'bs', 'company_prefix'
        ],
        "Financial": [
            'credit_card_number', 'credit_card_provider', 'credit_card_security_code',
            'credit_card_expire', 'currency', 'currency_code', 'currency_name',
            'pricetag', 'random_number', 'random_digit', 'random_digit_not_null'
        ],
        "Internet & Technology": [
            'ipv4', 'ipv6', 'mac_address', 'domain_name', 'domain_word', 'url',
            'user_name', 'user_agent', 'password', 'sha1', 'sha256', 'md5'
        ],
        "Text & Content": [
            'text', 'sentence', 'paragraph', 'word', 'words', 'sentences',
            'paragraphs', 'lexify', 'numerify', 'bothify'
        ],
        "Miscellaneous": [
            'uuid', 'random_element', 'random_elements', 'random_letter', 'random_letters',
            'random_lowercase_letter', 'random_uppercase_letter', 'random_int', 'random_float',
            'pyint', 'pyfloat', 'pystr', 'pylist', 'pytuple', 'pyset', 'pydict',
            'ssn', 'passport_number', 'license_plate'
        ]
    }
    
    # Filter to only include methods that actually exist
    available_categories = {}
    total_providers = 0
    
    for category, providers in categories.items():
        available_providers = []
        for provider in providers:
            if hasattr(faker, provider):
                available_providers.append(provider)
        
        if available_providers:
            available_categories[category] = sorted(available_providers)
            total_providers += len(available_providers)
    
    print(f"Found {total_providers} available Faker providers across {len(available_categories)} categories")
    return available_categories


def print_providers_by_category(categories: Dict[str, List[str]]) -> None:
    """Print providers organized by category."""
    print("\n" + "="*80)
    print("AVAILABLE FAKER PROVIDERS BY CATEGORY")
    print("="*80)
    
    for category, providers in categories.items():
        print(f"\n{category.upper()}:")
        print("-" * len(category))
        
        # Print providers in columns
        for i in range(0, len(providers), 3):
            chunk = providers[i:i+3]
            formatted_chunk = [f"{p:<25}" for p in chunk]
            print("  " + "  ".join(formatted_chunk))
        
        print(f"  ({len(providers)} providers)")


def print_all_providers_flat(categories: Dict[str, List[str]]) -> None:
    """Print all providers in a flat alphabetical list."""
    print("\n" + "="*80)
    print("ALL AVAILABLE FAKER PROVIDERS (ALPHABETICAL)")
    print("="*80)
    
    all_providers = []
    for providers in categories.values():
        all_providers.extend(providers)
    
    all_providers.sort()
    
    # Print in columns
    for i in range(0, len(all_providers), 4):
        chunk = all_providers[i:i+4]
        formatted_chunk = [f"{p:<20}" for p in chunk]
        print("  " + "  ".join(formatted_chunk))


def test_provider(provider_name: str) -> None:
    """Test a specific provider and show example output."""
    faker = Faker('en_US')
    
    if not hasattr(faker, provider_name):
        print(f"❌ Provider '{provider_name}' not found")
        return
    
    try:
        provider_method = getattr(faker, provider_name)
        result = provider_method()
        print(f"✅ {provider_name}: {result}")
    except Exception as e:
        print(f"❌ Error testing {provider_name}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="List all available Faker providers")
    parser.add_argument("--flat", action="store_true", help="Show flat alphabetical list only")
    parser.add_argument("--test", type=str, help="Test a specific provider")
    parser.add_argument("--count", action="store_true", help="Show only the count of providers")
    
    args = parser.parse_args()
    
    # Get all providers
    categories = get_all_faker_providers()
    
    if args.test:
        test_provider(args.test)
        return
    
    if args.count:
        total = sum(len(providers) for providers in categories.values())
        print(f"Total available Faker providers: {total}")
        return
    
    if args.flat:
        print_all_providers_flat(categories)
    else:
        print_providers_by_category(categories)
        print_all_providers_flat(categories)
    
    print(f"\n💡 Usage tips:")
    print(f"  - Use 'random_number' with 'digits' parameter for numeric IDs")
    print(f"  - Use 'date_of_birth' for birth dates with age constraints")
    print(f"  - Use 'random_element' with 'elements' array for categorical fields")
    print(f"  - Use 'email' for email addresses")
    print(f"  - Use 'phone_number' for phone numbers")
    print(f"  - Use 'uuid' for unique identifiers")


if __name__ == "__main__":
    import argparse
    main() 