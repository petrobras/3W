import os
from pathlib import Path
from typing import IO, Any


class ModelRecorder:
    @staticmethod
    def save_best_model(model: Any, filename: str | os.PathLike | IO[bytes]) -> None:
        """
        Save a model to disk depending on its type and file extension.
        Supports PyTorch and scikit-learn (Pickle).

        Parameters:
            model: Trained model object.
            filename: File name or file-like object where the model will be saved.
        """
        if isinstance(filename, (str, os.PathLike)):
            path = Path(filename)
            ext = path.suffix.lower()
        elif hasattr(filename, "write"):
            raise ValueError(
                f"Saving to file-like object '{filename}' is not supported. Please provide a valid file path."
            )
        else:
            raise TypeError(
                f"Invalid filename: `{filename}`. Expected a string, path-like object, or file-like object."
            )

        # PyTorch
        if ext in [".pt", ".pth"]:
            try:
                import torch

                torch.save(model.state_dict(), filename)
            except Exception as e:
                raise RuntimeError(f"Error saving PyTorch model: {e}")

        # scikit-learn or generic Python object
        elif ext in [".pkl", ".pickle"]:
            try:
                import pickle

                with open(filename, "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                raise RuntimeError(f"Error saving Pickle model: {e}")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def load_model(filename: str | os.PathLike | IO[bytes], model: Any = None) -> Any:
        """
        Load a model from disk depending on its type and file extension.
        Supports PyTorch (.pt, .pth) and scikit-learn/Pickle (.pkl, .pickle).

        Parameters:
            filename (str | os.PathLike | IO[bytes]):  Path or file-like object pointing to the saved model file.
            model (Any, optional):  An uninitialized model instance to load weights into. Required for PyTorch models.
        """
        if isinstance(filename, (str, os.PathLike)):
            path = Path(filename)
            ext = path.suffix.lower()
        elif hasattr(filename, "read"):
            raise ValueError(
                f"Loading from file-like object '{filename}' is not supported. Please provide a valid file path."
            )
        else:
            raise TypeError(
                f"Invalid filename: `{filename}`. Expected a string, path-like object, or file-like object."
            )

        # PyTorch
        if ext in [".pt", ".pth"]:
            try:
                import torch

                if model is None:
                    # Returns only the state_dict in case it does not have the model
                    return torch.load(filename)
                state_dict = torch.load(filename)
                model.load_state_dict(state_dict)
                return model
            except Exception as e:
                raise RuntimeError(f"Error loading PyTorch model: {e}")

        # scikit-learn or generic using Pickle
        elif ext in [".pkl", ".pickle"]:
            try:
                import pickle

                with open(filename, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Error loading Pickle model: {e}")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")
