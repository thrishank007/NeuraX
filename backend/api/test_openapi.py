"""
Test script to check OpenAPI schema generation
Run this to diagnose Swagger UI issues
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    print("Importing FastAPI app...")
    from backend.api.main import app
    print("✅ App imported successfully")
    
    print("\nGenerating OpenAPI schema...")
    schema = app.openapi()
    print("✅ OpenAPI schema generated successfully")
    
    print(f"\nSchema info:")
    print(f"  - Title: {schema.get('info', {}).get('title')}")
    print(f"  - Version: {schema.get('info', {}).get('version')}")
    print(f"  - Paths: {len(schema.get('paths', {}))}")
    
    print("\nAvailable endpoints:")
    for path, methods in schema.get('paths', {}).items():
        for method in methods.keys():
            print(f"  {method.upper()} {path}")
    
    print("\n✅ All checks passed! Swagger UI should work.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
