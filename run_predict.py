"""Entry point to run single-image prediction from project root."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import main
if __name__ == "__main__":
    main()
