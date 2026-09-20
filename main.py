from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.engines.da3.engine import DA3Engine
from app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    engine = DA3Engine()  # defaults to da3mono-large; see app/engines/da3/engine.py
    window = MainWindow(engine)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
