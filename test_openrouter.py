#!/usr/bin/env python3
"""Test OpenRouter API specifically"""

def test_openrouter():
    print("🔍 Testing OpenRouter API Configuration...")
    
    # Test OpenAI import
    try:
        from openai import OpenAI
        print("✅ OpenAI package available")
    except ImportError as e:
        print(f"❌ OpenAI package missing: {e}")
        print("💡 Install with: pip install openai==1.3.0")
        return False
    
    # Get API key
    api_key = input("🔑 Enter your OpenRouter API key (starts with sk-or-): ").strip()
    
    # Validate key format
    if not api_key.startswith('sk-or-'):
        print("⚠️  Warning: OpenRouter keys usually start with 'sk-or-'")
        print("   But continuing anyway...")
    
    if len(api_key) < 20:
        print("❌ API key seems too short")
        return False
    
    # Test client creation
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print("✅ OpenRouter client created successfully")
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False
    
    # Test API call
    try:
        print("🧪 Testing API call...")
        response = client.chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct:free",  # Free model for testing
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ API call successful!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        if "401" in str(e):
            print("💡 This usually means invalid API key")
        elif "insufficient_quota" in str(e):
            print("💡 This means no credits left on your account")
        elif "rate_limit" in str(e):
            print("💡 This means you're hitting rate limits")
        return False

if __name__ == "__main__":
    if test_openrouter():
        print("\n🎯 OpenRouter is working! Your API key is valid.")
        print("   The issue might be in the web app configuration.")
        print("   Try restarting the app and re-entering your key.")
    else:
        print("\n❌ OpenRouter test failed. Check your API key and credits.")