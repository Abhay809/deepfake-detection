"""Entry point to run evaluation from project root."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.evaluate import main
if __name__ == "__main__":
    main()
