"""Reuse the exact frozen v10 checks implementation."""
import sys
from candidates.ukl_native_sampling_v10 import checks as _frozen
sys.modules[__name__] = _frozen
