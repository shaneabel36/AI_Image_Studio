#!/usr/bin/env python3
"""Debug API configuration"""

import os
import sys

def test_openai_import():
    """Test if OpenAI can be imported"""
    try:
        from openai import OpenAI
        print("✅ OpenAI package imported successfully")
        return True
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False

def test_api_key_format(api_key):
    """Test if API key looks valid"""
    if not api_key:
        print("❌ No API key provided")
        return False
    
    if api_key == "your_openrouter_key_here":
        print("❌ Default placeholder API key detected")
        return False
    
    if len(api_key) < 10:
        print("❌ API key too short")
        return False
    
    print(f"✅ API key format looks valid (length: {len(api_key)})")
    return True

def test_client_creation(api_key):
    """Test creating OpenAI client"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        print("✅ OpenAI client created successfully")
        return client
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Debugging API Configuration...")
    
    # Test OpenAI import
    if not test_openai_import():
        print("\n💡 Install OpenAI: pip install openai==1.3.0")
        sys.exit(1)
    
    # Get API key from user
    api_key = input("\n🔑 Enter your API key to test: ").strip()
    
    # Test API key format
    if not test_api_key_format(api_key):
        sys.exit(1)
    
    # Test client creation
    client = test_client_creation(api_key)
    if client:
        print("\n🎯 API configuration should work!")
        print("   Make sure you entered this exact key in the web interface")
    else:
        print("\n❌ API configuration has issues")