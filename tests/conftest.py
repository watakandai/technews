import pathlib
import sys

# The package under test lives one level up, and the tests import it by name
# rather than requiring an install.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(name):
    """Bytes of a captured API response, for parser tests that never go online."""
    return (FIXTURES / name).read_bytes()
