# Data Validation Rule Generator

A Python tool that parses validation rules from text files and generates SQL INSERT statements for data validation using multiple LLM providers (OpenAI, Anthropic, Google Gemini, and Azure OpenAI).

## Features

- **Multi-LLM Support**: Uses OpenAI GPT, Anthropic Claude, Google Gemini, or Azure OpenAI for intelligent rule generation
- **Intelligent Field Extraction**: Automatically extracts field information from narrative text using LLM
- **Comprehensive Validation Schema**: Supports multiple rule types including NOT_NULL_CHECK, REGEX_VALIDATION, and more
- **Flexible Action Types**: Rules can FLAG, CORRECT, or REJECT invalid data
- **Type-Safe Implementation**: Proper type annotations and runtime checks for robust operation
- **Environment Variable Support**: Load API keys from `.env` file
- **Deduplication**: Automatically removes redundant rules to optimize token usage
- **Deterministic Mode**: Option for consistent rule generation across runs
- **Provider Agnostic**: Easy switching between different LLM providers

## Supported LLM Providers

- **OpenAI**: GPT-3.5-turbo, GPT-4, and other OpenAI models
- **Anthropic**: Claude models (claude-3-haiku, claude-3-sonnet, etc.)
- **Google Gemini**: Gemini Pro and other Google AI models
- **Azure OpenAI**: Azure-hosted OpenAI models

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with API keys:
   ```bash
   # For OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   
   # For Anthropic (optional)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Usage

### Basic Usage (OpenAI - Default)

```bash
python validation_rule_generator.py
```

### Using Different LLM Providers

#### OpenAI
```bash
python validation_rule_generator.py --provider openai --model gpt-3.5-turbo
```

#### Anthropic Claude
```bash
python validation_rule_generator.py --provider anthropic --model claude-3-haiku-20240307
```

### Advanced Configuration

```bash
python validation_rule_generator.py \
  --provider openai \
  --model gpt-4 \
  --temperature 0.1 \
  --max-tokens 1000 \
  --input-file config/validation_rules.txt \
  --output-file my_validation_rules.sql \
  --table-name validation_rules
```

### Command Line Options

- `--provider`: LLM provider (openai, anthropic, google, azure) - default: openai
- `--model`: Model name to use (defaults vary by provider)
- `--api-key`: API key (optional if set in .env file)
- `--temperature`: Temperature for LLM generation (default: 0.1)
- `--max-tokens`: Maximum tokens for LLM generation (default: 1000)
- `--input-file`: Input validation rules file (default: config/validation_rules.txt)
- `--output-file`: Output SQL file (default: generated_validation_rules.sql)
- `--table-name`: Target table name for INSERT statements (default: validation_rules)

## Programmatic Usage

### Basic Example

```python
from validation_rule_generator import ValidationRuleGenerator, LLMConfig

# Configure LLM
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.1
)

# Create generator
generator = ValidationRuleGenerator(llm_config=config)

# Generate SQL statements
sql_statements = generator.process_validation_rules("config/validation_rules.txt")

# Print results
for statement in sql_statements:
    print(statement)
```

### Multi-Provider Comparison

```python
from validation_rule_generator import ValidationRuleGenerator, LLMConfig

providers = [
    ("OpenAI", "openai", "gpt-4"),
    ("Anthropic", "anthropic", "claude-3-sonnet-20240229"),
    ("Google", "google", "gemini-pro")
]

for name, provider, model in providers:
    config = LLMConfig(provider=provider, model=model)
    generator = ValidationRuleGenerator(llm_config=config)
    sql_statements = generator.process_validation_rules("config/validation_rules.txt")
    print(f"{name}: Generated {len(sql_statements)} rules")
```

## Input Format

The tool supports two input formats:

### Simple Format (config/validation_rules.txt)

```
EDIPI (DoD ID):
The EDIPI must be a unique, 10-digit numeric value and is required for every record.

SSN:
The SSN must be a unique, 9-digit number in the format ###-##-####. This field is mandatory, and test or dummy values are not allowed.

Name (First, Last, Middle):
The full legal name is required. Each part (first, last, middle) must be alphabetic (letters only), can include hyphens or apostrophes, and must not exceed 100 characters.
```

### Complex Format (config/validation_rules_complex.txt)

```
Each record needs to have a unique EDIPI, which is a 10-digit number used as the DoD ID. It's a must for every entry.

The SSN also needs to be unique and exactly 9 digits long, written in the standard format like 123-45-6789. This field can't be left blank, and using fake or placeholder numbers isn't allowed.

When entering someone's name, all three parts—first, middle, and last—should be included. They should only contain letters, though it's okay to use hyphens or apostrophes. Just make sure no part of the name goes over 100 characters.
```

The LLM automatically extracts field information from both formats and generates appropriate validation rules.

## Output

The tool generates SQL INSERT statements for a validation rules table with the following schema:

```sql
-- Generated Validation Rules SQL Statements
-- Generated on: 2024-01-15 10:30:00
-- Generated using OPENAI gpt-4 for intelligent rule creation
-- Target table: validation_rules

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  1, 'edipi_required', 'edipi', 'NOT_NULL_CHECK', NULL,
  NULL, false, NULL, 'edipi IS NULL', 'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  2, 'edipi_numeric_format', 'edipi', 'REGEX_VALIDATION', '^[0-9]{10}$',
  NULL, false, NULL, 'edipi NOT RLIKE \'^[0-9]{10}$\'', 'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);
```

## Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI (required for default operation)
OPENAI_API_KEY=sk-your-openai-key

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Google Gemini (optional)
GOOGLE_API_KEY=your-google-api-key

# Azure OpenAI (optional)
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Examples

### Example Usage

```bash
# Generate rules using OpenAI GPT-4
python validation_rule_generator.py --provider openai --model gpt-4

# Generate rules using Anthropic Claude
python validation_rule_generator.py --provider anthropic --model claude-3-sonnet-20240229

# Generate rules using Google Gemini
python validation_rule_generator.py --provider google --model gemini-pro

# Custom input and output files
python validation_rule_generator.py --input-file my_rules.txt --output-file custom_rules.sql

# Different table name
python validation_rule_generator.py --table-name my_validation_rules
```

### Running Examples

```bash
# Run the example script to see different providers in action
python example_usage.py
```

## Error Handling

The tool includes comprehensive error handling for:
- Missing API keys
- Invalid provider configurations
- LLM API failures
- File I/O errors
- JSON parsing errors
- Type safety violations
- Network connectivity issues

## Recent Updates

- **Multi-LLM Support**: Added support for Google Gemini and Azure OpenAI
- **Enhanced Field Extraction**: LLM-powered extraction of field information from narrative text
- **Deterministic Mode**: Option for consistent rule generation across runs
- **Type Safety**: Added proper type annotations and runtime checks for robust operation
- **Client Validation**: Runtime checks ensure correct LLM client types are used
- **Token Optimization**: Automatic deduplication of redundant rules
- **Enhanced Error Messages**: More descriptive error messages for debugging
- **Flexible Input Formats**: Support for both structured and narrative input formats

## Dependencies

- `openai`: OpenAI API client
- `python-dotenv`: Environment variable loading
- `anthropic`: Anthropic Claude API client
- `google-generativeai`: Google Gemini API client
- `pandas`: Data manipulation and analysis
- `faker`: Test data generation
- `requests`: HTTP library for API calls
- `beautifulsoup4`: HTML/XML parsing

## License

This project is licensed under the MIT License.