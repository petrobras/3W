from pydantic import Field

from ..core.base_models import BaseModels, ModelsConfig


class MLPConfig(ModelsConfig):
    learning_rate: float = Field(..., lt=0.0, description="Learning rate must be < 0.")


class MLP(BaseModels):
    def __init__(self, config: MLPConfig):
        """
        LSTM model constructor.

        Args:
            config (LSTMConfig): Configuration for LSTM model.
        """
        if not isinstance(config, MLPConfig):
            raise TypeError("Expected LSTMConfig.")

        super().__init__(config)
