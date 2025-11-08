import pandas as pd 
from dataclasses import dataclass

df = pd.read_csv('swings.csv')

@dataclass
class Level:
    upper: float
    lower: float
    strength: int

