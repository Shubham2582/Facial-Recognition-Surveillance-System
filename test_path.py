# test_path.py
import os
from config.system_config import DATABASE_CONFIG

print(f"Database path from config: {DATABASE_CONFIG['database_dir']}")
print(f"Path exists: {os.path.exists(DATABASE_CONFIG['database_dir'])}")

# List directories if the path exists
if os.path.exists(DATABASE_CONFIG['database_dir']):
    print("Contents:")
    for item in os.listdir(DATABASE_CONFIG['database_dir']):
        print(f"  - {item}")