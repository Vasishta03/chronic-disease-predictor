import os
import sys
import subprocess
import time

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_command(command, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print(f"Python version {version.major}.{version.minor} is compatible")
    return True

def create_directories():
    directories = ['data', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Created directory: {directory}/")

def main():
    print_header("CHRONIC DISEASE PREDICTION SYSTEM SETUP")

    if not check_python_version():
        sys.exit(1)
    create_directories()

    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("\n  Trying alternative installation method...")
        if not run_command("python -m pip install -r requirements.txt", "Installing dependencies (alternative)"):
            sys.exit(1)

    if not run_command("python train_models.py", "Training ML models"):
        sys.exit(1)

    print("\n Verifying model files...")
    required_files = [
        'models/diabetes_model.pkl',
        'models/heart_disease_model.pkl', 
        'models/hypertension_model.pkl',
        'models/hypertension_scaler.pkl',
        'models/feature_names.pkl',
        'models/performance_metrics.pkl'
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n Warning: {len(missing_files)} model files are missing.")
    else:
        print("\nAll model files verified successfully!")

    print_header(" SETUP COMPLETED SUCCESSFULLY!")

    response = input("\n Would you like to launch the application now? (y/n): ").lower().strip()

    if response in ['y', 'yes']:
        print("\n Launching Chronic Disease Prediction System...")
        print("Opening in browser at http://localhost:8501")
        print("Press Ctrl+C to stop the application when finished.")

        try:
            subprocess.run("streamlit run app.py", shell=True, check=True)
        except KeyboardInterrupt:
            print("\n Application stopped.")
        except Exception as e:
            print(f"\nError launching application: {e}")
    else:
        print("\nSetup complete! Run 'streamlit run app.py' when ready to use the system.")

if __name__ == "__main__":
    main()
