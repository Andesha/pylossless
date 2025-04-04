import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--full-size", action="store", default=False, help="Use large dataset for tests"
    )