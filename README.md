# Data Validation Rule Generator

A Python tool that parses validation rules from text files and generates SQL INSERT statements for data validation using LLM providers (OpenAI and Anthropic).

## Features

- **LLM-Powered Rule Generation**: Uses OpenAI GPT or Anthropic Claude to intelligently generate validation rules based on field descriptions
- **Comprehensive Validation Schema**: Supports multiple rule types including NOT_NULL, REGEX_MATCH, FIXED_VALUE, DATE_NOT_FUTURE, LIST_MATCH, RANGE_CHECK, LENGTH_CHECK, LIST_TABLE_MATCH
- **Flexible Action Types**: Rules can FLAG, CORRECT, or REJECT invalid data
- **Type-Safe Implementation**: Proper type annotations and runtime checks for robust operation
- **Environment Variable Support**: Load API keys from `.env` file
- **Deduplication**: Automatically removes redundant rules to optimize token usage

## Supported LLM Providers

- **OpenAI**: GPT-3.5-turbo, GPT-4, and other OpenAI models
- **Anthropic**: Claude models (claude-3-haiku, claude-3-sonnet, etc.)

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

- `--provider`: LLM provider (openai, anthropic) - default: openai
- `--model`: Model name to use (defaults to gpt-3.5-turbo for OpenAI, claude-3-haiku-20240307 for Anthropic)
- `--api-key`: API key (optional if set in .env file)
- `--temperature`: Temperature for LLM generation (default: 0.1)
- `--max-tokens`: Maximum tokens for LLM generation (default: 1000)
- `--input-file`: Input validation rules file (default: config/validation_rules.txt)
- `--output-file`: Output SQL file (default: generated_validation_rules.sql)
- `--table-name`: Target table name for INSERT statements (default: validation_rules)

## Input Format

The tool expects a text file with field descriptions in the following format:

```
EDIPI (Electronic Data Interchange Personal Identifier): A unique 10-digit identifier assigned to each military service member. Must be numeric and exactly 10 digits long.

RANK: Military rank of the service member. Must be one of the valid ranks in the military hierarchy.

NAME: Full name of the service member. Must be non-empty and contain only letters, spaces, and hyphens.
```

## Output

The tool generates SQL INSERT statements for a validation rules table with the following schema:

```sql
-- Generated Validation Rules SQL Statements
-- Generated on: 2024-01-15 10:30:00
-- Generated using OPENAI gpt-3.5-turbo for intelligent rule creation
-- Target table: validation_rules

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  1, 'edipi_required', 'edipi', 'NOT_NULL', NULL,
  NULL, false, NULL, 'edipi IS NULL', 'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  2, 'edipi_numeric_format', 'edipi', 'REGEX_MATCH', '^[0-9]{10}$',
  NULL, false, NULL, 'edipi NOT RLIKE \'^[0-9]{10}$\'', 'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);
```

## Database Schema

The generated SQL statements are designed for a table with the following structure:

```sql
CREATE TABLE validation_rules (
  rule_id INT PRIMARY KEY,
  rule_name VARCHAR(255) NOT NULL,
  source_column VARCHAR(255) NOT NULL,
  rule_type VARCHAR(50) NOT NULL,
  rule_condition TEXT,
  compare_column_or_table VARCHAR(255),
  explode_flag BOOLEAN DEFAULT FALSE,
  threshold INT,
  expression_template TEXT,
  action_type VARCHAR(20) DEFAULT 'FLAG',
  is_enabled BOOLEAN DEFAULT TRUE,
  priority INT DEFAULT 1,
  rule_category VARCHAR(50) DEFAULT 'validation',
  created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI (required for default operation)
OPENAI_API_KEY=sk-your-openai-key

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

## Examples

### Example Usage

```bash
# Generate rules using OpenAI GPT-3.5-turbo
python validation_rule_generator.py

# Generate rules using Anthropic Claude
python validation_rule_generator.py --provider anthropic --model claude-3-haiku-20240307

# Custom input and output files
python validation_rule_generator.py --input-file my_rules.txt --output-file custom_rules.sql

# Different table name
python validation_rule_generator.py --table-name my_validation_rules
```

### Example Input File (config/validation_rules.txt)

```
EMAIL: Email address of the user. Must be a valid email format and not empty.

PHONE: Phone number in international format. Must start with + and contain only digits and hyphens.

AGE: Age of the person. Must be between 18 and 120 years old.

STATUS: Account status. Must be one of: active, inactive, suspended, pending.
```

## Error Handling

The tool includes comprehensive error handling for:
- Missing API keys
- Invalid provider configurations
- LLM API failures
- File I/O errors
- JSON parsing errors
- Type safety violations

## Recent Updates

- **Type Safety**: Added proper type annotations and runtime checks for robust operation
- **Client Validation**: Runtime checks ensure correct LLM client types are used
- **Token Optimization**: Automatic deduplication of redundant rules
- **Enhanced Error Messages**: More descriptive error messages for debugging

## Dependencies

- `openai`: OpenAI API client
- `python-dotenv`: Environment variable loading
- `anthropic`: Anthropic Claude API client (optional)

## License

This project is licensed under the MIT License.