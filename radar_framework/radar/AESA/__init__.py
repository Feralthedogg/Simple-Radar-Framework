# radar_framework/radar/AESA/__init__.py

"""
Top-level import for AESA radar module.
Exports:
  AESAParams: parameter dataclass
  AESA: core processing class
  JPDA, MHT: association engines
  AESAPipeline: end-to-end processing pipeline
"""
from .core.models import AESAParams, AESA
from .core.jpda import JPDA
from .core.mht import MHT
from .pipeline import AESAPipeline

__all__ = [
    'AESAParams', 'AESA',
    'JPDA', 'MHT',
    'AESAPipeline'
]
