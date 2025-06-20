#!/usr/bin/env python3
"""
Simple Faker Provider Web Search

This script uses OpenAI's web search to find the best Faker providers
for generating data based on field descriptions.
"""

import os
import sys
import json
import openai
from typing import Dict, Any


def find_faker_provider_with_web_search(field_description: str, api_key: str) -> Dict[str, Any]:
    """
    Use web search to find the best Faker provider for a field description.
    
    Args:
        field_description: Description of the field
        api_key: OpenAI API key
    
    Returns:
        Recommended Faker provider and parameters as a dict
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Create a focused prompt for Faker provider selection
        prompt = f"""
        Based on this field description: "{field_description}"
        
        Find the best Faker provider to generate realistic data for this field.
        Consider:
        1. The data type and format needed
        2. Realistic constraints and patterns
        3. Common use cases for this type of field
        
        Respond with ONLY a JSON object containing:
        {{
            "faker_provider": "provider_name",
            "parameters": {{"param1": "value1"}} or null,
            "reasoning": "brief explanation of why this provider is best"
        }}
        
        Use web search to find current best practices for data generation.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data generation expert. Always respond with valid JSON for Faker provider recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            tools=[{"type": "web_search"}],
            max_tokens=500,
            temperature=0.3
        )
        
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    result = json.loads(json_str)
                    return result
                else:
                    return {"error": "No JSON found in response", "raw_response": content}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": content}
        else:
            return {"error": "No response from OpenAI"}
            
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}


def main():
    """Main function."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("🤖 Faker Provider Web Search")
    print("=" * 50)
    print("This tool uses web search to find the best Faker providers for your field descriptions.")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            # Get field description from user
            field_desc = input("\n📝 Enter field description: ").strip()
            
            if field_desc.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not field_desc:
                continue
            
            print(f"\n🔍 Searching for best Faker provider for: {field_desc}")
            print("🤖 AI is thinking...")
            
            # Get recommendation
            result = find_faker_provider_with_web_search(field_desc, api_key)
            
            # Display result
            print("\n💡 Recommendation:")
            print("-" * 30)
            
            if isinstance(result, dict) and "error" in result:
                print(f"❌ Error: {result['error']}")
                if "raw_response" in result:
                    print(f"Raw response: {result['raw_response']}")
            elif isinstance(result, dict):
                print(f"✅ Faker Provider: {result.get('faker_provider', 'Unknown')}")
                
                params = result.get('parameters')
                if params:
                    print(f"📋 Parameters: {json.dumps(params, indent=2)}")
                else:
                    print("📋 Parameters: None")
                
                reasoning = result.get('reasoning', 'No reasoning provided')
                print(f"💭 Reasoning: {reasoning}")
            else:
                print(f"❌ Unexpected result: {result}")
            
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 