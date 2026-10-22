"""Render the current manuscript architecture from the released display script."""
from pathlib import Path
import subprocess, sys
if __name__ == '__main__':
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[2]/'reproduction/reproduce_main_displays.py'),'--figures','1'],check=True)
