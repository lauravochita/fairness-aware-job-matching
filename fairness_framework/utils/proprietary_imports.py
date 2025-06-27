"""
Proprietary imports module for fairness-aware job-matching framework.

This module handles imports from the proprietary job-matching system.
These components are not open-source and require access to the original system.

PROPRIETARY COMPONENTS:
- BiEncoderConfig: Configuration class for bi-encoder models
- BiEncoder: Bi-encoder model architecture
- evaluation_utils: FAISS search utilities (create_faiss_index, search_faiss)
- model utilities: numpy_serialize and related functions
"""

import sys
import os
from .config import Config


def load_proprietary_job_matching_classes():
    """
    Load BiEncoderConfig and BiEncoder from the proprietary job-matching system.

    Returns:
        tuple: (BiEncoderConfig, BiEncoder) classes

    Raises:
        ImportError: If proprietary system components are not available
    """
    try:
        # Add proprietary system path
        sys.path.append(Config.PROPRIETARY_INFERENCE_PATH)

        # Load types.py for BiEncoderConfig
        types_path = os.path.join(Config.PROPRIETARY_INFERENCE_PATH, "types.py")
        if not os.path.exists(types_path):
            raise FileNotFoundError(f"Proprietary types.py not found at {types_path}")

        with open(types_path, 'r') as f:
            types_content = f.read()

        types_namespace = {}
        exec(types_content, types_namespace)
        BiEncoderConfig = types_namespace['BiEncoderConfig']

        # Load utils.py for numpy_serialize function
        utils_path = os.path.join(Config.PROPRIETARY_INFERENCE_PATH, "utils.py")
        numpy_serialize = None
        if os.path.exists(utils_path):
            with open(utils_path, 'r') as f:
                utils_content = f.read()
            utils_namespace = {}
            exec(utils_content, utils_namespace)
            numpy_serialize = utils_namespace.get('numpy_serialize')

        # Load model.py with fixed imports
        model_path = os.path.join(Config.PROPRIETARY_INFERENCE_PATH, "model.py")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Proprietary model.py not found at {model_path}")

        with open(model_path, 'r') as f:
            model_content = f.read()

        # Fix import statements for execution
        fixed_model_content = model_content.replace(
            "from src.types import BiEncoderConfig",
            "# BiEncoderConfig injected"
        ).replace(
            "from src.utils import numpy_serialize",
            "# numpy_serialize injected"
        ).replace(
            "self.o_encoder",
            "self.p_encoder"
        )

        # Import required dependencies
        import torch
        import torch.nn.functional as torch_f
        from transformers import AutoModel, AutoTokenizer, BatchEncoding

        # Create execution namespace
        model_namespace = {
            'BiEncoderConfig': BiEncoderConfig,
            'numpy_serialize': numpy_serialize,
            'torch': torch,
            'torch_f': torch_f,
            'List': list,
            'Tensor': torch.Tensor,
            'nn': torch.nn,
            'AutoModel': AutoModel,
            'AutoTokenizer': AutoTokenizer,
            'BatchEncoding': BatchEncoding,
        }

        exec(fixed_model_content, model_namespace)
        BiEncoder = model_namespace['BiEncoder']

        return BiEncoderConfig, BiEncoder

    except Exception as e:
        raise ImportError(
            f"Failed to load proprietary job-matching system components: {e}\n"
            f"This requires access to the original Mysolution job-matching system.\n"
            f"See docs/proprietary_dependencies.md for more information."
        )


def load_fairws_vae_components():
    """
    Load VAE components from the FairWS framework.

    Returns:
        tuple: (VAE_fair, latent_loss) classes/functions
    """
    try:
        # Add FairWS path
        fairws_path = str(Config.BASE_DATA_PATH / "FairWS-main")
        if fairws_path not in sys.path:
            sys.path.append(fairws_path)

        # Import the original VAE components
        from VAE import VAE_fair, latent_loss

        return VAE_fair, latent_loss

    except ImportError as e:
        raise ImportError(
            f"Failed to load FairWS VAE components: {e}\n"
            f"This requires access to the FairWS framework VAE.py.\n"
            f"Expected path: {fairws_path}"
        )


def load_proprietary_evaluation_utils():
    """
    Load FAISS evaluation utilities from the proprietary system.

    Returns:
        tuple: (create_faiss_index, search_faiss) functions

    Raises:
        ImportError: If proprietary evaluation utilities are not available
    """
    try:
        # Add proprietary system path
        sys.path.append(Config.PROPRIETARY_SRC_PATH)

        # Import the evaluation utilities
        from tools.evaluation_utils import create_faiss_index, search_faiss

        return create_faiss_index, search_faiss

    except ImportError as e:
        raise ImportError(
            f"Failed to load proprietary FAISS evaluation utilities: {e}\n"
            f"This requires access to tools.evaluation_utils from the original system.\n"
            f"See docs/proprietary_dependencies.md for more information."
        )


def setup_proprietary_paths():
    """
    Add all necessary proprietary system paths to sys.path.
    Call this before importing any proprietary components.
    """
    paths_to_add = [
        Config.PROPRIETARY_SRC_PATH,
        Config.PROPRIETARY_INFERENCE_PATH
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)


# Convenience function for common import pattern
def get_proprietary_components():
    """
    Get all commonly used proprietary components in one call.

    Returns:
        dict: Dictionary containing all proprietary components
    """
    try:
        setup_proprietary_paths()
        BiEncoderConfig, BiEncoder = load_proprietary_job_matching_classes()
        create_faiss_index, search_faiss = load_proprietary_evaluation_utils()

        return {
            'BiEncoderConfig': BiEncoderConfig,
            'BiEncoder': BiEncoder,
            'create_faiss_index': create_faiss_index,
            'search_faiss': search_faiss
        }

    except ImportError as e:
        print(f"WARNING: Proprietary components not available: {e}")
        return None
