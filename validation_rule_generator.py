#!/usr/bin/env python3
"""
Enhanced Validation Rule Generator.

Parses validation rules from text file and generates SQL INSERT statements for data validation.
Uses LLM for intelligent rule generation with dynamic rule names, types, and templates.
LLM creates everything: rule_type, expression_template, rule_condition, and informative rule names.
"""

import re
import json
import argparse
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import os
from openai import OpenAI
from dotenv import load_dotenv

# Optional imports for different LLM providers
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM provider and model."""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


@dataclass
class ValidationRule:
    """Data class representing a validation rule with the new table schema."""
    rule_name: str
    source_column: str
    rule_type: str
    rule_condition: Optional[str] = None
    compare_column_or_table: Optional[str] = None
    explode_flag: bool = False
    threshold: Optional[int] = None
    expression_template: Optional[str] = None
    action_type: str = "FLAG"
    is_enabled: bool = True
    priority: int = 1
    rule_category: str = "validation"


@dataclass
class RuleTypeConfig:
    """Configuration for different rule types."""
    rule_type: str
    template: str
    suffix: str
    priority: int
    description: str
    requires_condition: bool = False


class ValidationRuleGenerator:
    """Generator class for creating validation rules using LLM."""

    # Define supported rule types - easily extensible
    SUPPORTED_RULE_TYPES = {
        'NOT_NULL_CHECK': RuleTypeConfig(
            rule_type='NOT_NULL_CHECK',
            template='',  # Will be generated as pure SQL logic
            suffix='null_check',
            priority=1,
            description='Check for required fields',
            requires_condition=False
        ),
        'REGEX_VALIDATION': RuleTypeConfig(
            rule_type='REGEX_VALIDATION', 
            template='',  # Will be generated as pure SQL logic
            suffix='format_check',
            priority=2,
            description='Pattern matching validation',
            requires_condition=True
        )
    }

    # Current active rule types (easily configurable) 
    ACTIVE_RULE_TYPES = ['NOT_NULL_CHECK', 'REGEX_VALIDATION']

    def __init__(self, llm_config: Optional[LLMConfig] = None, api_key: Optional[str] = None, deterministic: bool = True):
        """Initialize the generator with LLM configuration."""
        if llm_config is None:
            # Default to OpenAI if no config provided
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass api_key parameter.")
            
            llm_config = LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0.0 if deterministic else 0.1  # Use 0 temperature for deterministic results
            )
        
        self.llm_config = llm_config
        self.client: Any = self._initialize_llm_client()
        self.used_rule_names = set()  # Track unique rule names across all fields
        self.deterministic = deterministic

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.llm_config.provider.lower() == "openai":
            return OpenAI(api_key=self.llm_config.api_key)
        elif self.llm_config.provider.lower() == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic library not installed. Run: pip install anthropic")
            return Anthropic(api_key=self.llm_config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}. Supported: openai, anthropic")

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Make a call to the configured LLM provider."""
        if self.llm_config.provider.lower() == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI API")
            return content
        
        elif self.llm_config.provider.lower() == "anthropic":
            # Convert OpenAI format to Anthropic format
            prompt = self._convert_messages_to_anthropic_format(messages)
            response = self.client.messages.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens or 4000,
                temperature=self.llm_config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")

    def _convert_messages_to_anthropic_format(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to Anthropic format."""
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        return prompt

    def parse_validation_rules_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse validation rules from narrative text and extract field information using LLM."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Use LLM to extract field information from narrative text
        field_extraction_prompt = f"""
Analyze this validation requirements text and extract all data fields mentioned.
For each field, provide the field name and its validation requirements.

Text: {content}

Return a JSON array with objects containing:
- field_name: normalized field name (lowercase_with_underscores)
- original_name: original field name as mentioned in text
- description: complete validation requirements for this field
- suggested_column_names: array of possible column name variations

Example format:
[
  {{
    "field_name": "edipi",
    "original_name": "EDIPI", 
    "description": "Each record needs to have a unique EDIPI, which is a 10-digit number used as the DoD ID. It's a must for every entry.",
    "suggested_column_names": ["edipi", "dod_id", "edipi_number"]
  }}
]
"""

        messages = [
            {"role": "system", "content": "You are a data analyst expert at extracting field information from validation requirements."},
            {"role": "user", "content": field_extraction_prompt}
        ]

        try:
            llm_response = self._call_llm(messages)
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found in field extraction response")
            
            fields_data = json.loads(json_match.group())
            if not fields_data:
                raise ValueError("No fields extracted from input text")
            
            return fields_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse input file '{file_path}': {e}. Please check the input format and try again.") from e

    def _fallback_field_extraction(self, content: str) -> List[Dict[str, Any]]:
        """Removed fallback method - now raises error instead."""
        raise NotImplementedError("Fallback parsing has been removed. Please ensure input text is properly formatted for LLM parsing.")

    def generate_validation_rules_with_llm(self, field_info: Dict[str, Any]) -> List[ValidationRule]:
        """Use LLM to analyze requirements and generate rules based on active rule types."""
        
        # Create deterministic context for consistent rule generation
        field_context = self._create_deterministic_context(field_info)
        
        # Build dynamic rule type information for the prompt
        active_rules_info = []
        for rule_type in self.ACTIVE_RULE_TYPES:
            config = self.SUPPORTED_RULE_TYPES[rule_type]
            active_rules_info.append({
                'rule_type': config.rule_type,
                'suffix': config.suffix,
                'description': config.description,
                'requires_condition': config.requires_condition,
                'priority': config.priority
            })
        
        prompt = f"""
Analyze this field and determine appropriate validation rules based on the requirements.

Field: {field_info['field_name']} (Original: {field_info.get('original_name', field_info['field_name'])})
Requirements: {field_info['description']}

AVAILABLE RULE TYPES:
{self._format_rule_types_for_prompt(active_rules_info)}

RULE_CONDITION GENERATION GUIDELINES:
- For REGEX_VALIDATION: Generate the actual regex pattern (e.g., "^[0-9]{{10}}$" for 10-digit numbers)
- For NOT_NULL_CHECK: Set to null (no condition needed)

SQL EXPRESSION REQUIREMENTS:
- Generate ONLY the SQL logic/condition part (no SELECT statement)
- Expression should evaluate to TRUE when validation FAILS (not when it passes)
- Use {{source_column}} placeholder for the column name
- Use {{rule_condition}} placeholder when condition is needed
- Make expressions copy-pasteable into any SELECT statement
- For regex patterns, use proper SQL regex syntax for your database

SQL EXPRESSION EXAMPLES:
- NOT_NULL: "{{source_column}} IS NULL"
- REGEX: "({{source_column}} RLIKE '{{rule_condition}}')"

REQUIRED OUTPUT FORMAT (JSON Array):
[
  {{
    "rule_name": "field_name_suffix",
    "source_column": "field_name",
    "rule_type": "RULE_TYPE_NAME",
    "rule_condition": "condition_value_or_null",
    "compare_column_or_table": "table_or_column_name_or_null",
    "explode_flag": false,
    "threshold": null_or_number,
    "expression_template": "SQL_expression_with_placeholders",
    "action_type": "REJECT",
    "is_enabled": true,
    "priority": priority_number,
    "rule_category": "validation"
  }}
]

IMPORTANT: 
1. Generate 1-3 appropriate rules based on the field requirements
2. Ensure rule_condition is properly generated for rule types that require it
3. Use the exact JSON format above for easy parsing
4. Make rule names descriptive and follow snake_case convention
5. Set appropriate priorities (1=highest, 5=lowest)
"""

        messages = [
            {"role": "system", "content": f"You are a database validation expert. Analyze field requirements and generate appropriate validation rules. Context hash: {field_context}. Be consistent and deterministic. Always respond with valid JSON array."},
            {"role": "user", "content": prompt}
        ]

        try:
            llm_response = self._call_llm(messages)
            return self._parse_llm_response(field_info['field_name'], llm_response)

        except Exception as e:
            raise RuntimeError(f"LLM API call failed for field {field_info['field_name']}: {e}") from e

    def _format_rule_types_for_prompt(self, active_rules_info: List[Dict]) -> str:
        """Format rule types information for the prompt in a compact way."""
        formatted = []
        for rule_info in active_rules_info:
            condition_note = " (requires rule_condition)" if rule_info['requires_condition'] else " (rule_condition=null)"
            formatted.append(f"- {rule_info['rule_type']}: {rule_info['description']}{condition_note}")
            formatted.append(f"  └─ Suffix: {rule_info['suffix']}, Priority: {rule_info['priority']}")
        return "\n".join(formatted)

    def _parse_llm_response(self, field_name: str, llm_response: str) -> List[ValidationRule]:
        """Parse the LLM response and convert to ValidationRule objects."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found in LLM response")

            rules_data = json.loads(json_match.group())
            if not rules_data:
                raise ValueError("No rules generated from LLM response")
            
            rules = []

            for rule_data in rules_data:
                # Get rule type configuration
                rule_type = rule_data.get('rule_type')
                if rule_type not in self.SUPPORTED_RULE_TYPES:
                    raise ValueError(f"Unknown rule type '{rule_type}' in LLM response")
                
                rule_config = self.SUPPORTED_RULE_TYPES[rule_type]
                
                # Ensure rule name follows pattern
                expected_rule_name = f"{field_name}_{rule_config.suffix}"
                rule_name = rule_data.get('rule_name', expected_rule_name)
                rule_name = self._ensure_unique_rule_name(rule_name)
                
                # Get expression template from LLM response
                expression_template = rule_data.get('expression_template')
                if not expression_template:
                    raise ValueError(f"Missing expression_template for rule '{rule_name}' of type '{rule_type}'")
                
                # Validate rule condition requirements
                rule_condition = rule_data.get('rule_condition')
                if rule_config.requires_condition and not rule_condition:
                    raise ValueError(f"Rule type {rule_type} requires condition but none provided for {field_name}")
                
                # Convert string "null" to actual None
                if rule_condition == "null":
                    rule_condition = None
                
                rule = ValidationRule(
                    rule_name=rule_name,
                    source_column=rule_data.get('source_column', field_name),
                    rule_type=rule_type,
                    rule_condition=rule_condition,
                    compare_column_or_table=rule_data.get('compare_column_or_table'),
                    explode_flag=rule_data.get('explode_flag', False),
                    threshold=rule_data.get('threshold'),
                    expression_template=expression_template,
                    action_type=rule_data.get('action_type', 'REJECT'),
                    is_enabled=rule_data.get('is_enabled', True),
                    priority=rule_data.get('priority', rule_config.priority),
                    rule_category=rule_data.get('rule_category', 'validation')
                )
                
                # Validate the rule
                rule = self._validate_rule(rule)
                rules.append(rule)
                
                # Track the rule name
                self.used_rule_names.add(rule.rule_name)

            return rules

        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response for field '{field_name}': {e}") from e

    def _create_deterministic_context(self, field_info: Dict[str, Any]) -> str:
        """Create a deterministic context hash for consistent rule generation."""
        import hashlib
        
        # Create a consistent string from field info
        context_string = f"{field_info['field_name']}|{field_info['description']}"
        
        # Add sorted suggested column names for consistency
        if 'suggested_column_names' in field_info:
            sorted_columns = sorted(field_info['suggested_column_names'])
            context_string += f"|{','.join(sorted_columns)}"
        
        # Create a short hash for deterministic seeding
        context_hash = hashlib.md5(context_string.encode()).hexdigest()[:8]
        return context_hash

    def _ensure_unique_rule_name(self, proposed_name: str, max_length: int = 50) -> str:
        """Ensure rule name is unique, follows snake_case, and within length limit."""
        # Convert to snake_case if not already
        snake_case_name = self._to_snake_case(proposed_name)
        
        # Truncate if too long
        if len(snake_case_name) > max_length:
            snake_case_name = snake_case_name[:max_length].rstrip('_')
        
        # For deterministic naming, we don't modify the name unless there's a real conflict
        # The LLM should generate consistent names, so conflicts should be rare
        if snake_case_name not in self.used_rule_names:
            return snake_case_name
        
        # Only add suffix if there's an actual conflict (shouldn't happen with deterministic generation)
        counter = 1
        base_name = snake_case_name[:max_length-4]  # Leave room for suffix
        while f"{base_name}_{counter}" in self.used_rule_names:
            counter += 1
        
        print(f"Warning: Rule name conflict resolved: {snake_case_name} -> {base_name}_{counter}")
        return f"{base_name}_{counter}"

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case format."""
        # Replace spaces and hyphens with underscores
        snake = re.sub(r'[-\s]+', '_', text)
        
        # Convert camelCase to snake_case
        snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', snake)
        
        # Convert to lowercase
        snake = snake.lower()
        
        # Remove any non-alphanumeric characters except underscores
        snake = re.sub(r'[^a-z0-9_]', '', snake)
        
        # Remove multiple consecutive underscores
        snake = re.sub(r'_+', '_', snake)
        
        # Remove leading/trailing underscores
        snake = snake.strip('_')
        
        return snake

    def _validate_rule(self, rule: ValidationRule) -> ValidationRule:
        """Validate and fix common issues in generated rules."""
        
        # Ensure expression template exists
        if not rule.expression_template:
            rule.expression_template = f"{rule.source_column} IS NULL"
        
        # Ensure rule type is not empty
        if not rule.rule_type:
            rule.rule_type = "VALIDATION_CHECK"
        
        # Normalize source column
        rule.source_column = self._normalize_column_name(rule.source_column)
        
        return rule

    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column name to lowercase with underscores."""
        normalized = column_name.lower().strip()
        normalized = re.sub(r'\s+', '_', normalized)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized

    def generate_sql_insert_statements(self, rules: List[ValidationRule], table_name: str = "validation_rules") -> List[str]:
        """Generate SQL INSERT statements for the validation rules."""
        sql_statements = []

        for i, rule in enumerate(rules, start=1):
            # Escape single quotes in string values
            def escape_sql_string(value):
                if value is None:
                    return "NULL"
                return "'" + str(value).replace("'", "''") + "'"
            
            # Format all values
            rule_name_escaped = escape_sql_string(rule.rule_name)
            source_column_escaped = escape_sql_string(rule.source_column)
            rule_type_escaped = escape_sql_string(rule.rule_type)
            rule_condition_escaped = escape_sql_string(rule.rule_condition)
            compare_column_escaped = escape_sql_string(rule.compare_column_or_table)
            threshold_value = str(rule.threshold) if rule.threshold is not None else "NULL"
            expression_template_escaped = escape_sql_string(rule.expression_template)
            action_type_escaped = escape_sql_string(rule.action_type)
            rule_category_escaped = escape_sql_string(rule.rule_category)
            
            sql = f"""INSERT INTO {table_name} (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  {i}, {rule_name_escaped}, {source_column_escaped}, {rule_type_escaped}, {rule_condition_escaped},
  {compare_column_escaped}, {str(rule.explode_flag).lower()}, {threshold_value}, {expression_template_escaped},
  {action_type_escaped}, {str(rule.is_enabled).lower()}, {rule.priority}, {rule_category_escaped}, 
  current_timestamp(), current_timestamp()
);"""
            sql_statements.append(sql)

        return sql_statements

    def process_validation_rules(self, file_path: str, table_name: str = "validation_rules") -> List[str]:
        """Main method to process validation rules file and generate SQL statements."""
        print(f"Processing validation rules from: {file_path}")
        print(f"Using {self.llm_config.provider.upper()} {self.llm_config.model}")
        print(f"Active rule types: {', '.join(self.ACTIVE_RULE_TYPES)}")

        # Parse the file using LLM
        field_info_list = self.parse_validation_rules_file(file_path)
        print(f"Extracted {len(field_info_list)} fields from text")

        all_rules = []

        # Generate rules for each field using LLM
        for field_info in field_info_list:
            print(f"Processing field: {field_info['field_name']} ({field_info.get('original_name', 'N/A')})")
            try:
                rules = self.generate_validation_rules_with_llm(field_info)
                all_rules.extend(rules)
                print(f"Generated {len(rules)} rules for {field_info['field_name']}")
                
                # Show generated rule names and conditions
                for rule in rules:
                    condition_info = f" (condition: {rule.rule_condition})" if rule.rule_condition else ""
                    print(f"  └─ {rule.rule_name} [{rule.rule_type}]{condition_info}")
                
            except (ValueError, RuntimeError) as e:
                print(f"Error processing {field_info['field_name']}: {e}")
                continue

        # Generate SQL statements
        sql_statements = self.generate_sql_insert_statements(all_rules, table_name)

        print(f"\nTotal rules generated: {len(all_rules)}")
        print(f"Unique rule names: {len(self.used_rule_names)}")
        
        return sql_statements


def main():
    """Main function to run the validation rule generator."""
    parser = argparse.ArgumentParser(description='Generate SQL validation rules from narrative text using LLM intelligence')
    parser.add_argument('--api-key', type=str, help='API key (optional if set in .env file)')
    parser.add_argument('--provider', type=str, default='openai', 
                       choices=['openai', 'anthropic'],
                       help='LLM provider to use')
    parser.add_argument('--model', type=str, help='Model name to use')
    parser.add_argument('--rule-types', type=str, nargs='+', 
                       default=['NOT_NULL_CHECK', 'REGEX_VALIDATION'],
                       help='Active rule types to generate (default: NOT_NULL_CHECK REGEX_VALIDATION)')
    parser.add_argument('--list-rule-types', action='store_true',
                       help='List all supported rule types and exit')
    parser.add_argument('--deterministic', action='store_true', default=True, 
                       help='Generate deterministic rule names (default: True)')
    parser.add_argument('--temperature', type=float, default=None, 
                       help='Temperature for LLM generation (default: 0.0 for deterministic, 0.1 for creative)')
    parser.add_argument('--max-tokens', type=int, default=2000, help='Maximum tokens for LLM generation')
    parser.add_argument('--input-file', type=str, default='config/validation_rules.txt', help='Input validation requirements file')
    parser.add_argument('--output-file', type=str, default='generated_validation_rules.sql', help='Output SQL file')
    parser.add_argument('--table-name', type=str, default='validation_rules', help='Target table name for INSERT statements')

    args = parser.parse_args()

    # Handle list rule types
    if args.list_rule_types:
        generator = ValidationRuleGenerator()
        print("Supported Rule Types:")
        for rule_type, config in generator.SUPPORTED_RULE_TYPES.items():
            condition_note = " (requires condition)" if config.requires_condition else ""
            print(f"  {rule_type}: {config.description}{condition_note}")
            print(f"    - Suffix: {config.suffix}")
            print(f"    - Priority: {config.priority}")
            print(f"    - Template: {config.template}")
            print()
        exit(0)

    # Set default models for each provider
    default_models = {
        'openai': 'gpt-3.5-turbo',
        'anthropic': 'claude-3-haiku-20240307'
    }

    try:
        # Create LLM configuration
        temperature = args.temperature
        if temperature is None:
            temperature = 0.0 if args.deterministic else 0.1
            
        llm_config = LLMConfig(
            provider=args.provider,
            model=args.model or default_models[args.provider],
            api_key=args.api_key,
            temperature=temperature,
            max_tokens=args.max_tokens
        )

        # Initialize the generator
        generator = ValidationRuleGenerator(llm_config=llm_config, deterministic=args.deterministic)
        
        # Override active rule types if specified
        if args.rule_types:
            # Validate that all specified rule types are supported
            unsupported = [rt for rt in args.rule_types if rt not in generator.SUPPORTED_RULE_TYPES]
            if unsupported:
                print(f"Error: Unsupported rule types: {unsupported}")
                print(f"Supported types: {list(generator.SUPPORTED_RULE_TYPES.keys())}")
                print("Use --list-rule-types to see all available rule types")
                exit(1)
            generator.ACTIVE_RULE_TYPES = args.rule_types
            print(f"Using custom rule types: {args.rule_types}")

        # Process the validation rules file
        sql_statements = generator.process_validation_rules(args.input_file, args.table_name)

        # Write SQL statements to file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("-- Generated Validation Rules SQL Statements\n")
            f.write("-- Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(f"-- Generated using {llm_config.provider.upper()} {llm_config.model} with intelligent rule creation\n")
            f.write(f"-- Target table: {args.table_name}\n")
            f.write(f"-- Total rules: {len(sql_statements)}\n")
            f.write("\n")

            for sql in sql_statements:
                f.write(sql + "\n\n")

        print(f"\nGenerated {len(sql_statements)} SQL statements")
        print(f"Output written to: {args.output_file}")

        # Print preview
        print("\nPreview of generated SQL statements:")
        for i, sql in enumerate(sql_statements[:3]):
            print(f"\n--- Rule {i+1} ---")
            print(sql)

    except (ValueError, RuntimeError, FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()