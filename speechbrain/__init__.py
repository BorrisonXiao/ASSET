"""SpeechBrain runtime subset used by ASSET (Apache-2.0)."""
from .core import Brain, Stage, create_experiment_directory
from .utils.run_opts import RunOptions
from .utils.importutils import lazy_export_all
from pathlib import Path
__version__ = version = Path(__file__).with_name("version.txt").read_text().strip()
parse_arguments = RunOptions.from_command_line_args
lazy_export_all(__file__, __name__, export_subpackages=True)
