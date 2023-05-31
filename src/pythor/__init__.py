### Import useful objects, check module dependencies import

### New Pipeline of incremental learning
from .base.workflow import iterative_tabular_benchmark_base
from .utils.stats import strategy_metrics
from .utils.linalg import cross_correlation_mtx, tail_corr
from .utils.ml import save_thor_model, load_thor_model

## Logging
import logging
import sys

# file_handler = logging.FileHandler(filename='tmp.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
# handlers = [file_handler, stdout_handler]
handlers = [stdout_handler]

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(handlers=handlers, level=logging.INFO, format=FORMAT)
