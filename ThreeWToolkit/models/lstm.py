from pydantic import Field

from ..core.base_models import BaseModels, ModelsConfig

class LSTMConfig(ModelsConfig):
    learning_rate: float = Field(..., gt = 0.0, description="Learning rate must be > 0.")

class LSTM(BaseModels):
    def __init__(self, config: LSTMConfig):
        """
        LSTM model constructor.

        Args:
            config (LSTMConfig): Configuration for LSTM model.
        """
        if not isinstance(config, LSTMConfig):
            raise TypeError("Expected LSTMConfig.")
        
        super().__init__(config)

