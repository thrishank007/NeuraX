#!/usr/bin/env python3
"""
Setup HuggingFace authentication for accessing gated models
"""
import os
import sys
from pathlib import Path
from huggingface_hub import login, whoami
from loguru import logger
import getpass

def check_current_auth():
    """Check if already authenticated with HuggingFace"""
    try:
        user_info = whoami()
        logger.info(f"✅ Already authenticated with HuggingFace as: {user_info['name']}")
        return True, user_info['name']
    except Exception as e:
        logger.info("❌ Not currently authenticated with HuggingFace")
        return False, None

def setup_environment_variable():
    """Guide user to set up HF_TOKEN environment variable"""
    print("\n" + "="*60)
    print("Setting up HuggingFace Token as Environment Variable")
    print("="*60)
    
    print("\n1. Get your HuggingFace token:")
    print("   - Go to: https://huggingface.co/settings/tokens")
    print("   - Create a new token or copy existing one")
    print("   - Make sure it has 'Read' permissions")
    
    print("\n2. For gated models (like Llama, MiniCPM-V), also:")
    print("   - Visit the model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)")
    print("   - Click 'Request access' and wait for approval")
    
    token = getpass.getpass("\nEnter your HuggingFace token (hidden input): ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return False
    
    # Test the token
    try:
        login(token=token)
        user_info = whoami()
        print(f"✅ Token is valid! Authenticated as: {user_info['name']}")
        
        # Show how to set environment variable
        print("\n" + "="*60)
        print("How to set HF_TOKEN environment variable:")
        print("="*60)
        
        print("\nFor Windows (PowerShell):")
        print(f'$env:HF_TOKEN = "{token}"')
        print("# Or permanently:")
        print(f'[Environment]::SetEnvironmentVariable("HF_TOKEN", "{token}", "User")')
        
        print("\nFor Windows (Command Prompt):")
        print(f'set HF_TOKEN={token}')
        print("# Or permanently:")
        print(f'setx HF_TOKEN "{token}"')
        
        print("\nFor Linux/macOS:")
        print(f'export HF_TOKEN="{token}"')
        print("# Add to ~/.bashrc or ~/.zshrc for permanent setup:")
        print(f'echo \'export HF_TOKEN="{token}"\' >> ~/.bashrc')
        
        print("\n⚠️  Security Note: Keep your token secure and don't share it!")
        
        return True
        
    except Exception as e:
        print(f"❌ Token validation failed: {e}")
        return False

def interactive_login():
    """Use huggingface_hub's interactive login"""
    print("\n" + "="*60)
    print("Interactive HuggingFace Login")
    print("="*60)
    
    print("\nThis will open a browser for authentication.")
    print("If you prefer to set up manually, use the environment variable option.")
    
    try:
        login()
        user_info = whoami()
        print(f"✅ Successfully authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Interactive login failed: {e}")
        return False

def test_model_access():
    """Test access to gated models"""
    print("\n" + "="*60)
    print("Testing Model Access")
    print("="*60)
    
    gated_models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "openbmb/MiniCPM-V-2_6",
        "Qwen/Qwen2-VL-7B-Instruct"
    ]
    
    from huggingface_hub import model_info
    
    for model_name in gated_models:
        try:
            info = model_info(model_name)
            print(f"✅ Access granted to: {model_name}")
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e):
                print(f"❌ No access to: {model_name} (request access on HuggingFace)")
            else:
                print(f"⚠️  Could not check: {model_name} ({e})")

def main():
    """Main function"""
    print("HuggingFace Authentication Setup")
    print("="*40)
    
    # Check current authentication
    is_auth, username = check_current_auth()
    
    if is_auth:
        print(f"\nYou're already authenticated as: {username}")
        test_access = input("Test access to gated models? (y/N): ").strip().lower()
        if test_access == 'y':
            test_model_access()
        return 0
    
    print("\nChoose authentication method:")
    print("1. Set up environment variable (recommended)")
    print("2. Interactive browser login")
    print("3. Skip setup")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    success = False
    if choice == '1':
        success = setup_environment_variable()
    elif choice == '2':
        success = interactive_login()
    elif choice == '3':
        print("Skipping authentication setup.")
        print("Note: Some models may not work without authentication.")
        return 0
    else:
        print("Invalid choice. Exiting.")
        return 1
    
    if success:
        print("\n✅ Authentication setup complete!")
        test_access = input("Test access to gated models? (y/N): ").strip().lower()
        if test_access == 'y':
            test_model_access()
        
        print("\n📝 Next steps:")
        print("1. Restart your terminal/IDE to load new environment variables")
        print("2. Run 'python test_llama_integration.py' to test model loading")
        print("3. Use 'python main_launcher.py' to start the application")
        return 0
    else:
        print("\n❌ Authentication setup failed.")
        print("Please try again or check your token/network connection.")
        return 1

if __name__ == "__main__":
    sys.exit(main())