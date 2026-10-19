import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # lets tests share fixtures such as test_predict.PixelDetector
