"""
    The main driver script
"""

from __future__ import print_function
import sys
from model import evaluate_model


if len(sys.argv) != 1:
    print ("The correct syntax for running the script is python main.py")
    evaluate_model()    
