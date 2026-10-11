"""Reuse the exact frozen v10 sampling implementation."""
import sys
from candidates.ukl_native_sampling_v10 import sampling as _frozen
sys.modules[__name__] = _frozen
