from abc import ABC
from pydantic import BaseModel

class FeatureExtractorConfig(BaseModel):
    pass

class BaseFeatureExtractor(ABC):
    def __init__(self, config: FeatureExtractorConfig):
        self.config = config