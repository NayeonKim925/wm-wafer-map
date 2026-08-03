"""Top-level package for the WM-811K wafer-map defect classifier.

Sub-packages
------------
- :mod:`src.config`     - typed configuration (dataclasses + YAML).
- :mod:`src.data`       - dataset acquisition, cleaning and batching.
- :mod:`src.models`     - model architectures and a build registry.
- :mod:`src.training`   - the training loop.
- :mod:`src.evaluation` - metrics and the evaluator.
- :mod:`src.inference`  - the predictor for new wafer maps.
- :mod:`src.utils`      - logging, seeding, device and path helpers.
"""

__version__ = "0.1.0"
