# Data Validation Test Data Generator

A tool to help you find the best Faker provider for your data fields using OpenAI's web search capabilities.

## What it does

- Given a field description, uses OpenAI (with web search) to recommend the best Faker provider and parameters for generating realistic test data.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Find the best Faker provider for your field
```bash
python scripts/simple_faker_web_search.py
```
You will be prompted to enter a field description (e.g., "A unique 8-digit numeric identifier for each employee."). The tool will use web search to recommend the most appropriate Faker provider, parameters, and reasoning.

Type `quit` to exit.

## Example

```
📝 Enter field description: Valid email address in format user@domain.com.

🔍 Searching for best Faker provider for: Valid email address in format user@domain.com.
🤖 AI is thinking...

💡 Recommendation:
------------------------------
✅ Faker Provider: email
📋 Parameters: None
💭 Reasoning: The 'email' provider generates realistic email addresses in the correct format, matching the field description and common data generation practices.
------------------------------
```

## Requirements
- Python 3.8+
- OpenAI API key
- pandas, faker, openai, python-dotenv

## Project Structure

```
data_validation_script/
├── config/              # Configuration files
├── data/                # Generated data
├── output/              # Output files
├── scripts/             # Command line tools
│   └── simple_faker_web_search.py  # Main tool for finding Faker providers
├── src/                 # Source code
└── README.md            # This file
```

## License

MIT License