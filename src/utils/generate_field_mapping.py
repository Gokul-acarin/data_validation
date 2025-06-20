#!/usr/bin/env python3
"""
Generate field mappings using OpenAI API.

This script reads field descriptions and uses OpenAI to generate appropriate
Faker provider mappings for each field.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import openai
from faker import Faker


def get_all_faker_providers() -> List[str]:
    """Get all available Faker provider methods."""
    faker = Faker('en_US')
    
    
    # Common providers by category
    name_providers = [
        'name', 'first_name', 'last_name', 'full_name', 'name_male', 'name_female',
        'prefix', 'prefix_male', 'prefix_female', 'suffix', 'suffix_male', 'suffix_female'
    ]
    
    address_providers = [
        'address', 'street_address', 'street_name', 'street_suffix', 'city', 'state',
        'state_abbr', 'country', 'country_code', 'postcode', 'zipcode', 'latitude', 'longitude'
    ]
    
    contact_providers = [
        'email', 'safe_email', 'free_email', 'company_email', 'phone_number', 'cellphone',
        'phone_number', 'country_calling_code'
    ]
    
    personal_providers = [
        'date_of_birth', 'date', 'time', 'year', 'month', 'day_of_week', 'day_of_month',
        'timezone', 'ssn', 'passport_number', 'license_plate'
    ]
    
    company_providers = [
        'company', 'company_suffix', 'job', 'job_title', 'company_name', 'catch_phrase',
        'bs', 'company_prefix'
    ]
    
    financial_providers = [
        'credit_card_number', 'credit_card_provider', 'credit_card_security_code',
        'credit_card_expire', 'currency', 'currency_code', 'currency_name',
        'pricetag', 'random_number', 'random_digit', 'random_digit_not_null'
    ]
    
    internet_providers = [
        'ipv4', 'ipv6', 'mac_address', 'domain_name', 'domain_word', 'url',
        'user_name', 'user_agent', 'password', 'sha1', 'sha256', 'md5'
    ]
    
    text_providers = [
        'text', 'sentence', 'paragraph', 'word', 'words', 'sentence', 'sentences',
        'paragraph', 'paragraphs', 'lexify', 'numerify', 'bothify'
    ]
    
    misc_providers = [
        'uuid', 'random_element', 'random_elements', 'random_letter', 'random_letters',
        'random_lowercase_letter', 'random_uppercase_letter', 'random_int', 'random_float',
        'pyint', 'pyfloat', 'pystr', 'pylist', 'pytuple', 'pyset', 'pydict'
    ]
    
    # Combine all providers
    all_providers = (
        name_providers + address_providers + contact_providers + personal_providers +
        company_providers + financial_providers + internet_providers + text_providers + misc_providers
    )
    
    # Filter to only include methods that actually exist on the Faker instance
    available_providers = []
    for provider in all_providers:
        if hasattr(faker, provider):
            available_providers.append(provider)
    
    # Sort alphabetically
    available_providers.sort()
    
    print(f"Found {len(available_providers)} available Faker providers")
    return available_providers


def create_mapping_prompt(field_descriptions: List[Dict[str, str]], faker_providers: List[str]) -> str:
    """Create a prompt for OpenAI to generate field mappings."""
    
    # Group providers by category for better organization
    provider_categories = {
        "Names & Personal Info": [p for p in faker_providers if any(keyword in p for keyword in ['name', 'first', 'last', 'full', 'prefix', 'suffix'])],
        "Address & Location": [p for p in faker_providers if any(keyword in p for keyword in ['address', 'street', 'city', 'state', 'country', 'postcode', 'zip', 'lat', 'long'])],
        "Contact Information": [p for p in faker_providers if any(keyword in p for keyword in ['email', 'phone', 'cell'])],
        "Dates & Time": [p for p in faker_providers if any(keyword in p for keyword in ['date', 'time', 'year', 'month', 'day'])],
        "Company & Business": [p for p in faker_providers if any(keyword in p for keyword in ['company', 'job', 'business'])],
        "Financial": [p for p in faker_providers if any(keyword in p for keyword in ['credit', 'currency', 'price', 'random_number'])],
        "Internet & Technology": [p for p in faker_providers if any(keyword in p for keyword in ['ip', 'mac', 'domain', 'url', 'user', 'password'])],
        "Text & Content": [p for p in faker_providers if any(keyword in p for keyword in ['text', 'sentence', 'paragraph', 'word'])],
        "Miscellaneous": [p for p in faker_providers if not any(keyword in p for keyword in ['name', 'address', 'email', 'phone', 'date', 'company', 'credit', 'ip', 'text'])]
    }
    
    prompt = """You are a data engineering expert. Given the following field descriptions, 
map each field to the most appropriate Faker provider and any additional parameters.

For each field, provide:
1. faker_provider: The Faker provider method from the list below
2. parameters: Any parameters needed for the provider (e.g., locale, format, etc.)
3. validation_rules: Basic validation rules for the field

Available Faker providers by category:
"""
    
    for category, providers in provider_categories.items():
        if providers:
            prompt += f"\n{category}:\n"
            # Format providers in columns for better readability
            for i in range(0, len(providers), 4):
                chunk = providers[i:i+4]
                prompt += f"  {', '.join(chunk)}\n"
    
    prompt += f"\nField descriptions:\n"
    for i, field in enumerate(field_descriptions, 1):
        prompt += f"{i}. {field['field_name']}: {field['field_description']}\n"
    
    prompt += """
Please respond with a JSON array where each object has:
{
  "field_name": "string",
  "faker_provider": "string", 
  "parameters": "object or null",
  "validation_rules": "string"
}

Important guidelines:
- Choose the most specific and appropriate provider for each field
- Use 'random_number' with 'digits' parameter for numeric IDs
- Use 'date_of_birth' for birth dates with age constraints
- Use 'random_element' with 'elements' array for categorical fields
- Use 'email' for email addresses
- Use 'phone_number' for phone numbers
- Use 'uuid' for unique identifiers when appropriate

Example response:
[
  {
    "field_name": "Employee ID",
    "faker_provider": "random_number",
    "parameters": {"digits": 8, "fix_len": true},
    "validation_rules": "8-digit numeric identifier"
  },
  {
    "field_name": "Email Address", 
    "faker_provider": "email",
    "parameters": null,
    "validation_rules": "Valid email format"
  },
  {
    "field_name": "Gender",
    "faker_provider": "random_element",
    "parameters": {"elements": ["M", "F", "X", "U"]},
    "validation_rules": "M, F, X, or U"
  }
]
"""
    
    return prompt


def call_openai_api(prompt: str, api_key: str, model: str = "gpt-4o", enable_web_search: bool = True) -> str:
    """Call OpenAI API to generate field mappings with optional web search."""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a data engineering expert specializing in test data generation. Always respond with valid JSON arrays. When web search is enabled, use it to find the most up-to-date information about data validation best practices and field mapping strategies."},
            {"role": "user", "content": prompt}
        ]
        
        # Prepare tools for web search if enabled
        tools = None
        if enable_web_search:
            tools = [
                {
                    "type": "web_search"
                }
            ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=1500,
            temperature=0.3
        )
        
        if not response.choices:
            raise Exception("No response from OpenAI API")
        
        content = response.choices[0].message.content
        if not content:
            raise Exception("Empty response from OpenAI API")
        
        print(f"OpenAI API call successful (model: {model}, web_search: {enable_web_search})")
        return content
        
    except openai.AuthenticationError:
        raise Exception("Invalid OpenAI API key")
    except openai.RateLimitError:
        raise Exception("OpenAI rate limit exceeded")
    except openai.APIError as e:
        raise Exception(f"OpenAI API error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error calling OpenAI API: {e}")


def parse_openai_response(response: str) -> List[Dict[str, str]]:
    """Parse the OpenAI API response to extract field mappings."""
    try:
        # Try to extract JSON from the response
        # Look for JSON array in the response
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise Exception("No JSON array found in OpenAI response")
        
        json_str = response[start_idx:end_idx + 1]
        mappings = json.loads(json_str)
        
        if not isinstance(mappings, list):
            raise Exception("OpenAI response is not a list")
        
        # Validate each mapping
        for i, mapping in enumerate(mappings):
            required_fields = ['field_name', 'faker_provider', 'validation_rules']
            for field in required_fields:
                if field not in mapping:
                    raise Exception(f"Missing required field '{field}' in mapping {i}")
            
            # Ensure parameters is present (can be null)
            if 'parameters' not in mapping:
                mapping['parameters'] = None
        
        print(f"Successfully parsed {len(mappings)} field mappings from OpenAI response")
        return mappings
        
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON in OpenAI response: {e}")
    except Exception as e:
        raise Exception(f"Failed to parse OpenAI response: {e}")


def generate_field_mappings(descriptions_file: Path, output_file: Path, api_key: str, model: str = "gpt-4o", enable_web_search: bool = True) -> None:
    """Generate field mappings using OpenAI API with optional web search."""
    print(f"Generating field mappings...")
    print(f"Input file: {descriptions_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Get all available Faker providers
        faker_providers = get_all_faker_providers()
        
        # Read field descriptions
        field_descriptions = []
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                field_descriptions.append(row)
        
        if not field_descriptions:
            raise Exception("No field descriptions found in input file")
        
        print(f"Read {len(field_descriptions)} field descriptions")
        
        # Create prompt with all available providers
        prompt = create_mapping_prompt(field_descriptions, faker_providers)
        
        # Call OpenAI API
        response = call_openai_api(prompt, api_key, model, enable_web_search)
        
        # Parse response
        mappings = parse_openai_response(response)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write mappings to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['field_name', 'faker_provider', 'parameters', 'validation_rules'])
            
            for mapping in mappings:
                parameters_str = json.dumps(mapping['parameters']) if mapping['parameters'] else ''
                writer.writerow([
                    mapping['field_name'],
                    mapping['faker_provider'],
                    parameters_str,
                    mapping['validation_rules']
                ])
        
        print(f"✅ Successfully generated {len(mappings)} field mappings")
        print(f"Output saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: Descriptions file not found: {descriptions_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating field mappings: {e}")
        sys.exit(1)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate field mappings using OpenAI API")
    parser.add_argument("descriptions_file", help="Path to field descriptions CSV file")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search functionality")
    parser.add_argument("--output", help="Output file path (default: output/mappings/field_mappings.csv)")
    
    args = parser.parse_args()
    
    # Check for API key
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    descriptions_file = Path(args.descriptions_file)
    output_file = Path(args.output) if args.output else Path("output/mappings/field_mappings.csv")
    enable_web_search = not args.no_web_search
    
    if not descriptions_file.exists():
        print(f"❌ Error: Descriptions file not found: {descriptions_file}")
        sys.exit(1)
    
    print(f"🤖 Using model: {args.model}")
    print(f"🌐 Web search: {'Enabled' if enable_web_search else 'Disabled'}")
    
    generate_field_mappings(descriptions_file, output_file, api_key, args.model, enable_web_search)


if __name__ == "__main__":
    main() 