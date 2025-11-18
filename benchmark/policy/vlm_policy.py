"""VLM policy placeholder.

TODO: Implement VLM policy wrapper.
"""

import numpy as np
from typing import Dict, Any, Optional

from benchmark.policy.base import BasePolicy


class VLMPolicy(BasePolicy):
    """Placeholder for VLM (Vision-Language Model) policy.
    
    TODO: Implement when VLM policy is ready.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        raise NotImplementedError("VLM policy not yet implemented")
    
    def predict_action(
        self, 
        observation: Dict[str, np.ndarray],
        goal: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        raise NotImplementedError("VLM policy not yet implemented")
    
    def reset(self):
        pass
    
    def load_checkpoint(self, checkpoint_path: str):
        raise NotImplementedError("VLM policy not yet implemented")
    
    @property
    def name(self) -> str:
        return "vlm"
