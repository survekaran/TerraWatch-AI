from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Phase 2" in at.title[0].value
