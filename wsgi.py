import sys
import os

# Add the path to your application here
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set the application module (app) - change `your_app` to the name of your Flask application
from app import app as application

if __name__ == "__main__":
    application.run()  # Optional: for local testing
