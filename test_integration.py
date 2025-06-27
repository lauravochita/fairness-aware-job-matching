import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import torch
    import numpy as np
    import pandas as pd

    print("Basic imports work")

    from fairness_framework.utils.config import Config

    print("Config import works")

    from fairness_framework.utils.proprietary_imports import load_fairws_vae_components

    VAE_fair, latent_loss = load_fairws_vae_components()
    print("FairWS components loaded")

except Exception as e:
    print(f"Error: {e}")