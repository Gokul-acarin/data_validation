#!/usr/bin/env python3
"""
Validation Rule Generator.

Parses validation rules from text file and generates SQL INSERT statements for data validation.
Uses LLM for intelligent rule generation based on field descriptions.
Updated to match the new table schema with rule_type, expression_template, etc.
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


class ValidationRuleGenerator:
    """Generator class for creating validation rules using LLM."""

    # Define supported rule types - easily extensible
    SUPPORTED_RULE_TYPES = {
        'NOT_NULL': RuleTypeConfig(
            rule_type='NOT_NULL',
            template='{source_column} IS NULL',
            suffix='not_null',
            priority=1,
            description='Check for required fields'
        ),
        'REGEX_MATCH': RuleTypeConfig(
            rule_type='REGEX_MATCH', 
            template="NOT ({source_column} RLIKE '{rule_condition}')",
            suffix='format_check',
            priority=2,
            description='Pattern matching validation'
        )
    }

    # Current active rule types (easily configurable)
    ACTIVE_RULE_TYPES = ['NOT_NULL', 'REGEX_MATCH']

    def __init__(self, llm_config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        """Initialize the generator with LLM configuration."""
        if llm_config is None:
            # Default to OpenAI if no config provided
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass api_key parameter.")
            
            llm_config = LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                api_key=api_key
            )
        
        self.llm_config = llm_config
        self.client: Any = self._initialize_llm_client()

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
            # Ensure we have an OpenAI client
            if not hasattr(self.client, 'chat'):
                raise ValueError("Expected OpenAI client but got different client type")
            
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
            # Ensure we have an Anthropic client
            if not hasattr(self.client, 'messages'):
                raise ValueError("Expected Anthropic client but got different client type")
            
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

    def parse_validation_rules_file(self, file_path: str) -> List[Dict[str, str]]:
        """Parse the validation rules text file and extract field information."""
        rules = []

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split by field sections (field name followed by colon)
        sections = re.split(r'\n(?=[A-Z][^:\n]*:)', content)

        for section in sections:
            if not section.strip():
                continue

            lines = section.strip().split('\n')
            if len(lines) < 2:
                continue

            # Extract field name (remove colon and parentheses content)
            field_line = lines[0]
            field_name = field_line.split(':')[0].strip()
            field_name = re.sub(r'\([^)]*\)', '', field_name).strip()

            # Extract description (everything after the first line)
            description = ' '.join(lines[1:]).strip()

            rules.append({
                'field_name': field_name,
                'description': description
            })

        return rules

    def generate_validation_rules_with_llm(self, field_info: Dict[str, str]) -> List[ValidationRule]:
        """Use LLM to generate validation rules for a field."""
        prompt = self._create_validation_prompt(field_info)

        try:
            # Build system message dynamically based on active rule types
            active_types = ", ".join(self.ACTIVE_RULE_TYPES)
            system_msg = f"You are a data validation expert. Generate exactly {len(self.ACTIVE_RULE_TYPES)} rules per field: {active_types} validation."
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
            
            llm_response = self._call_llm(messages)
            
            if llm_response is None:
                raise ValueError("Empty response from LLM API")
            
            return self._parse_llm_response(field_info['field_name'], llm_response)

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

    def _create_validation_prompt(self, field_info: Dict[str, str]) -> str:
        """Create a scalable prompt based on active rule types."""
        normalized_field = self._normalize_column_name(field_info['field_name'])
        
        # Build rule type descriptions dynamically
        active_rules = [self.SUPPORTED_RULE_TYPES[rule_type] for rule_type in self.ACTIVE_RULE_TYPES]
        rule_descriptions = []
        rule_examples = []
        
        for i, rule_config in enumerate(active_rules, 1):
            rule_descriptions.append(f"{i}. {rule_config.rule_type}: {rule_config.description}")
            
            # Generate example based on rule type
            if rule_config.rule_type == 'NOT_NULL':
                example = {
                    "rule_name": f"{normalized_field}_{rule_config.suffix}",
                    "source_column": normalized_field,
                    "rule_type": rule_config.rule_type,
                    "rule_condition": None,
                    "compare_column_or_table": None,
                    "explode_flag": False,
                    "threshold": None,
                    "expression_template": rule_config.template,
                    "action_type": "REJECT",
                    "is_enabled": True,
                    "priority": rule_config.priority,
                    "rule_category": "validation"
                }
            elif rule_config.rule_type == 'REGEX_MATCH':
                example = {
                    "rule_name": f"{normalized_field}_{rule_config.suffix}",
                    "source_column": normalized_field,
                    "rule_type": rule_config.rule_type,
                    "rule_condition": "appropriate_regex_pattern",
                    "compare_column_or_table": None,
                    "explode_flag": False,
                    "threshold": None,
                    "expression_template": rule_config.template,
                    "action_type": "REJECT",
                    "is_enabled": True,
                    "priority": rule_config.priority,
                    "rule_category": "validation"
                }
            # Future rule types can add their examples here
            
            rule_examples.append(example)
        
        return f"""Generate exactly {len(self.ACTIVE_RULE_TYPES)} validation rules for: {field_info['field_name']}
Description: {field_info['description']}

GENERATE THESE RULE TYPES:
{chr(10).join(rule_descriptions)}

CRITICAL FORMATTING RULES:
- source_column: MUST be lowercase with underscores (e.g., email_address, pay_grade)
- rule_name: Use pattern: {normalized_field}_[rule_suffix]
- expression_template: Required for all rules

RULE TEMPLATES:
{chr(10).join([f"- {rule.rule_type}: {rule.template}" for rule in active_rules])}

Return JSON array with exactly {len(self.ACTIVE_RULE_TYPES)} objects:
{json.dumps(rule_examples, indent=2)}

Generate appropriate patterns/conditions based on the field description."""

    def _parse_llm_response(self, original_field_name: str, llm_response: str) -> List[ValidationRule]:
        """Parse the LLM response and convert to ValidationRule objects."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found in LLM response")

            rules_data = json.loads(json_match.group())
            rules = []

            for rule_data in rules_data:
                # Normalize source_column to lowercase with underscores
                source_column = rule_data.get('source_column', original_field_name.lower())
                source_column = self._normalize_column_name(source_column)
                
                rule = ValidationRule(
                    rule_name=rule_data.get('rule_name', ''),
                    source_column=source_column,
                    rule_type=rule_data.get('rule_type', 'NOT_NULL'),
                    rule_condition=rule_data.get('rule_condition'),
                    compare_column_or_table=rule_data.get('compare_column_or_table'),
                    explode_flag=rule_data.get('explode_flag', False),
                    threshold=rule_data.get('threshold'),
                    expression_template=rule_data.get('expression_template'),
                    action_type=rule_data.get('action_type', 'FLAG'),
                    is_enabled=rule_data.get('is_enabled', True),
                    priority=rule_data.get('priority', 1),
                    rule_category=rule_data.get('rule_category', 'validation')
                )
                
                # Validate and fix the rule
                rule = self._validate_and_fix_rule(rule)
                rules.append(rule)

            return rules

        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response: {e}") from e

    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column name to lowercase with underscores."""
        # Convert to lowercase and replace spaces with underscores
        normalized = column_name.lower().strip()
        normalized = re.sub(r'\s+', '_', normalized)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized

    def _validate_and_fix_rule(self, rule: ValidationRule) -> ValidationRule:
        """Validate and fix common issues in generated rules - scalable for all rule types."""
        
        # Fix missing expression_template using configured templates
        if not rule.expression_template:
            rule.expression_template = self._generate_expression_template(rule)
        
        # Ensure rule_name uses proper pattern based on rule type
        if not rule.rule_name or rule.rule_name in ['', 'undefined']:
            rule_config = self.SUPPORTED_RULE_TYPES.get(rule.rule_type)
            if rule_config:
                rule.rule_name = f"{rule.source_column}_{rule_config.suffix}"
            else:
                # Fallback for unknown rule types
                rule.rule_name = f"{rule.source_column}_{rule.rule_type.lower()}"
        
        # Apply rule-type specific fixes
        rule = self._apply_rule_type_fixes(rule)
        
        return rule

    def _apply_rule_type_fixes(self, rule: ValidationRule) -> ValidationRule:
        """Apply specific fixes based on rule type - easily extensible."""
        
        if rule.rule_type == 'REGEX_MATCH':
            # Fix REGEX_MATCH expression template logic
            if rule.expression_template and 'NOT REGEXP_LIKE' in rule.expression_template:
                rule.expression_template = "NOT ({source_column} RLIKE '{rule_condition}')"
            elif rule.expression_template and 'REGEXP_LIKE' not in rule.expression_template and 'RLIKE' not in rule.expression_template:
                rule.expression_template = "NOT ({source_column} RLIKE '{rule_condition}')"
        
        elif rule.rule_type == 'LENGTH_CHECK':
            # Fix LENGTH_CHECK rule_condition format (extract just the number)
            if rule.rule_condition:
                match = re.search(r'\d+', str(rule.rule_condition))
                if match:
                    rule.rule_condition = match.group()
        
        elif rule.rule_type in ['RANGE_CHECK', 'DATE_COMPARISON']:
            # Fix field reference issues for comparison rules
            if rule.rule_condition and not rule.compare_column_or_table:
                if any(col in str(rule.rule_condition) for col in ['_date', '_time', 'hire', 'start', 'end']):
                    rule.compare_column_or_table = rule.rule_condition
                    rule.rule_condition = None
        
        # Additional rule type fixes can be added here as needed
        
        return rule

    def _generate_expression_template(self, rule: ValidationRule) -> str:
        """Generate appropriate expression template - uses configured templates."""
        rule_config = self.SUPPORTED_RULE_TYPES.get(rule.rule_type)
        if rule_config:
            return rule_config.template
        else:
            # Fallback for unknown rule types
            return '{source_column} IS NOT NULL'

    def generate_sql_insert_statements(self, rules: List[ValidationRule], table_name: str = "validation_rules") -> List[str]:
        """Generate SQL INSERT statements for the validation rules."""
        sql_statements = []

        for i, rule in enumerate(rules, start=1):
            # Escape single quotes in string values to prevent SQL injection
            def escape_sql_string(value):
                if value is None:
                    return "NULL"
                return "'" + str(value).replace("'", "''") + "'"
            
            # Format all string values with proper escaping
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
        print(f"Rules per field: {len(self.ACTIVE_RULE_TYPES)}")

        # Parse the file
        field_info_list = self.parse_validation_rules_file(file_path)
        print(f"Found {len(field_info_list)} fields to process")

        all_rules = []
        used_rule_names = set()  # Track unique rule names

        # Generate rules for each field using LLM
        for field_info in field_info_list:
            print(f"Processing field: {field_info['field_name']}")
            try:
                rules = self.generate_validation_rules_with_llm(field_info)
                
                # Validate rule count matches active rule types
                if len(rules) != len(self.ACTIVE_RULE_TYPES):
                    print(f"Warning: Expected {len(self.ACTIVE_RULE_TYPES)} rules, got {len(rules)} for field {field_info['field_name']}")
                
                # Check for uniqueness (should be rare with full field names)
                for rule in rules:
                    if rule.rule_name in used_rule_names:
                        print(f"Warning: Duplicate rule name '{rule.rule_name}' detected")
                    used_rule_names.add(rule.rule_name)
                
                all_rules.extend(rules)
                print(f"Generated {len(rules)} rules for {field_info['field_name']}")
            except (ValueError, RuntimeError) as e:
                print(f"Error processing {field_info['field_name']}: {e}")
                continue

        # Generate SQL statements
        sql_statements = self.generate_sql_insert_statements(all_rules, table_name)

        return sql_statements


def main():
    """Main function to run the validation rule generator."""
    parser = argparse.ArgumentParser(description='Generate SQL validation rules from text descriptions using LLM')
    parser.add_argument('--api-key', type=str, help='API key (optional if set in .env file)')
    parser.add_argument('--provider', type=str, default='openai', 
                       choices=['openai', 'anthropic'],
                       help='LLM provider to use')
    parser.add_argument('--model', type=str, help='Model name to use')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for LLM generation')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens for LLM generation')
    parser.add_argument('--input-file', type=str, default='config/validation_rules.txt', help='Input validation rules file')
    parser.add_argument('--output-file', type=str, default='generated_validation_rules.sql', help='Output SQL file')
    parser.add_argument('--table-name', type=str, default='validation_rules', help='Target table name for INSERT statements')
    parser.add_argument('--rule-types', type=str, nargs='+', 
                       default=['NOT_NULL', 'REGEX_MATCH'],
                       help='Active rule types to generate (default: NOT_NULL REGEX_MATCH)')

    args = parser.parse_args()

    # Set default models for each provider (optimized for token efficiency)
    default_models = {
        'openai': 'gpt-3.5-turbo',
        'anthropic': 'claude-3-haiku-20240307'
    }

    try:
        # Create LLM configuration (token-optimized by default)
        llm_config = LLMConfig(
            provider=args.provider,
            model=args.model or default_models[args.provider],
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

        # Initialize the generator
        generator = ValidationRuleGenerator(llm_config=llm_config)
        
        # Override active rule types if specified
        if args.rule_types:
            # Validate that all specified rule types are supported
            unsupported = [rt for rt in args.rule_types if rt not in generator.SUPPORTED_RULE_TYPES]
            if unsupported:
                print(f"Error: Unsupported rule types: {unsupported}")
                print(f"Supported types: {list(generator.SUPPORTED_RULE_TYPES.keys())}")
                exit(1)
            generator.ACTIVE_RULE_TYPES = args.rule_types
            print(f"Using custom rule types: {args.rule_types}")

        # Process the validation rules file
        sql_statements = generator.process_validation_rules(args.input_file, args.table_name)

        # Write SQL statements to file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("-- Generated Validation Rules SQL Statements\n")
            f.write("-- Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(f"-- Generated using {llm_config.provider.upper()} {llm_config.model} for intelligent rule creation\n")
            f.write(f"-- Target table: {args.table_name}\n")
            f.write("\n")

            for sql in sql_statements:
                f.write(sql + "\n\n")

        print(f"\nGenerated {len(sql_statements)} SQL statements")
        print(f"Output written to: {args.output_file}")

        # Also print first few statements as preview
        print("\nPreview of generated SQL statements:")
        for i, sql in enumerate(sql_statements[:2]):
            print(f"\n--- Rule {i+1} ---")
            print(sql)

    except (ValueError, RuntimeError, FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}")
        print("Make sure you have set the appropriate API key in your .env file or provided --api-key parameter")
        exit(1)


if __name__ == "__main__":
    main()