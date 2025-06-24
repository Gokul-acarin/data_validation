#!/usr/bin/env python3
"""
Example usage of the LLM-agnostic validation rule generator.

This script demonstrates how to use different LLM providers
with the validation rule generator.
"""

from validation_rule_generator import ValidationRuleGenerator, LLMConfig


def example_openai_usage():
    """Example using OpenAI GPT-4."""
    print("=== OpenAI GPT-4 Example ===")
    
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.1
    )
    
    generator = ValidationRuleGenerator(llm_config=config)
    sql_statements = generator.process_validation_rules("config/validation_rules.txt")
    
    print(f"Generated {len(sql_statements)} SQL statements with OpenAI")
    return sql_statements


def example_anthropic_usage():
    """Example using Anthropic Claude."""
    print("\n=== Anthropic Claude Example ===")
    
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        temperature=0.1
    )
    
    generator = ValidationRuleGenerator(llm_config=config)
    sql_statements = generator.process_validation_rules("config/validation_rules.txt")
    
    print(f"Generated {len(sql_statements)} SQL statements with Claude")
    return sql_statements


def example_google_usage():
    """Example using Google Gemini."""
    print("\n=== Google Gemini Example ===")
    
    config = LLMConfig(
        provider="google",
        model="gemini-pro",
        temperature=0.1
    )
    
    generator = ValidationRuleGenerator(llm_config=config)
    sql_statements = generator.process_validation_rules("config/validation_rules.txt")
    
    print(f"Generated {len(sql_statements)} SQL statements with Gemini")
    return sql_statements


def example_azure_usage():
    """Example using Azure OpenAI."""
    print("\n=== Azure OpenAI Example ===")
    
    config = LLMConfig(
        provider="azure",
        model="gpt-4",
        base_url="https://your-resource.openai.azure.com/",
        temperature=0.1
    )
    
    generator = ValidationRuleGenerator(llm_config=config)
    sql_statements = generator.process_validation_rules("config/validation_rules.txt")
    
    print(f"Generated {len(sql_statements)} SQL statements with Azure OpenAI")
    return sql_statements


def compare_providers():
    """Compare results from different providers."""
    print("\n=== Provider Comparison ===")
    
    providers = [
        ("OpenAI", "openai", "gpt-4"),
        ("Anthropic", "anthropic", "claude-3-sonnet-20240229"),
        ("Google", "google", "gemini-pro")
    ]
    
    results = {}
    
    for name, provider, model in providers:
        try:
            config = LLMConfig(
                provider=provider,
                model=model,
                temperature=0.1
            )
            
            generator = ValidationRuleGenerator(llm_config=config)
            sql_statements = generator.process_validation_rules("config/validation_rules.txt")
            results[name] = len(sql_statements)
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = 0
    
    print("\nResults comparison:")
    for name, count in results.items():
        print(f"{name}: {count} rules generated")


if __name__ == "__main__":
    print("LLM-Agnostic Validation Rule Generator Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run
    # Make sure you have the appropriate API keys in your .env file
    
    # example_openai_usage()
    # example_anthropic_usage()
    # example_google_usage()
    # example_azure_usage()
    
    # Compare different providers
    # compare_providers()
    
    print("\nTo run examples, uncomment the desired function calls above.")
    print("Make sure you have the appropriate API keys in your .env file:")
    print("- OPENAI_API_KEY for OpenAI and Azure")
    print("- ANTHROPIC_API_KEY for Anthropic")
    print("- GOOGLE_API_KEY for Google") 