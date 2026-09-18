from argparse import ArgumentParser
from pathlib import Path
import sys

import vedo
from PyQt5.QtWidgets import QApplication, QMessageBox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.views import MainWindow


DEFAULT_SCAN = PROJECT_ROOT / "data/own/meshes/Lay3/full.obj"
DEFAULT_SCAN = ""


def parse_args():
    parser = ArgumentParser(description="Fit a STAR body model to a single 3D scan.")
    parser.add_argument(
        "scan",
        nargs="?",
        default=str(DEFAULT_SCAN),
        help="path to an OBJ, PLY, or STL scan",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output mesh path (default: out/<scan_name>.ply)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vedo.settings.default_backend = "vtk"
    app = QApplication(sys.argv)
    try:
        window = MainWindow(args.scan, args.output)
    except Exception as error:
        QMessageBox.critical(None, "Could not load scan", str(error))
        return 1
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
