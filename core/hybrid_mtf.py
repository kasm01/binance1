"""
Backward-compat shim.

Projede asıl implementasyon: models/hybrid_mtf.py
Bazı eski kodlar `core.hybrid_mtf` import edebileceği için burada re-export yapıyoruz.

NOT:
- Buraya iş mantığı eklemeyin.
- Tek kaynak: models.hybrid_mtf
"""

from __future__ import annotations

import logging

system_logger = logging.getLogger("system")

try:
    # canonical location
    from models.hybrid_mtf import HybridMTF, MultiTimeframeHybridEnsemble  # noqa: F401
except Exception as e:
    # Import zamanında log düşsün ama crash etmeyelim (özellikle tooling sırasında).
    system_logger.exception("[core.hybrid_mtf] Failed to import from models.hybrid_mtf: %s", e)
    raise

__all__ = ["HybridMTF", "MultiTimeframeHybridEnsemble"]
