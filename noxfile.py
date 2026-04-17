"""Nox configuration for megan_global_explanations package."""

import nox

# Python versions to test against
PYTHON_VERSIONS = ['3.10', '3.11', '3.12']


@nox.session(python=PYTHON_VERSIONS, venv_backend='uv')
def test(session):
    """Run unit tests with pytest."""
    session.install('.')
    session.install('pytest')
    session.install('lorem-text')
    session.run(
        'pytest',
        'tests/',
        '-v',
        '--ignore=tests/test_gpt.py',  # Requires OpenAI API key
    )


@nox.session(python=PYTHON_VERSIONS, venv_backend='uv')
def install_check(session):
    """Check that the package can be installed and imported."""
    session.install('.')
    session.run('python', '-c', "import megan_global_explanations; print('Import successful')")
