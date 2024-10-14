import os
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from typing import Dict, Tuple, List, Optional


class CONFIG(BaseModel):
    MODEL_PATH: str = Field(default=None)

    @classmethod
    def from_yaml(cls, config_file: str = 'config.yaml'):
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)


# Initialize the configuration
config_path =  os.path.join(str(os.getcwd()), 'cfg', 'config.yaml')
conf = CONFIG.from_yaml(config_path)

if __name__ == "__main__":
    pass
    # print(f"Model path: {conf.MODEL_PATH}")

