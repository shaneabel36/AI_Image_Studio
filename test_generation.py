#!/usr/bin/env python3
"""Test image generation to debug issues"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from app import workflow_automator

def test_generation():
    print("🔍 Testing AI Image Studio Configuration...")
    
    # Check if workflow is configured
    if workflow_automator is None:
        print("❌ Workflow automator is None - API key not configured properly")
        return False
    
    print("✅ Workflow automator exists")
    print(f"   API Key: {'Set' if workflow_automator.api_key else 'Not set'}")
    print(f"   Base prompt: {workflow_automator.base_prompt}")
    print(f"   Model: {workflow_automator.gen_model}")
    
    # Check directories
    print(f"   Base directory: {workflow_automator.base_dir}")
    
    # Test directory creation
    try:
        for name, path in workflow_automator.dirs.items():
            os.makedirs(path, exist_ok=True)
            print(f"   ✅ {name}: {path}")
    except Exception as e:
        print(f"   ❌ Directory creation failed: {e}")
        return False
    
    print("\n🎯 Configuration looks good!")
    print("   Try generating an image through the web interface")
    return True

if __name__ == "__main__":
    test_generation()