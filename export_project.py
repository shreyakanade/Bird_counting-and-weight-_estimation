import zipfile
import os

def zip_project():
    zip_name = "poultry_submission.zip"
    files_to_include = ['main.py', 'processor.py', 'generate_report.py', 'requirements.txt', 'README.md']
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file}")
        
        # Include empty outputs folder
        if not os.path.exists('outputs'): os.makedirs('outputs')
        zipf.writestr('outputs/.keep', '')

    print(f"\n✅ Project successfully zipped into: {zip_name}")

if __name__ == "__main__":
    zip_project()