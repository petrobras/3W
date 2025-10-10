import json
import requests  # type: ignore
import hashlib

from tqdm import tqdm
from pathlib import Path

from ..utils.general_utils import GeneralUtils
from typing import List
from pydantic import BaseModel, field_validator

FIGSHARE_BASE_URL = "https://api.figshare.com/v2"
FIGSHARE_VERSION_IDS = {
    "1.0.0": "29205926",
    "1.1.0": "29205932",
    "1.1.1": "29205947",
    "2.0.0": "29205836",
}


class GetFigshareDataValidator(BaseModel):
    """Validator for figshare data download parameters.

    Ensures that the download path exists, is a directory, the version
    is supported, and the chunk size is positive.

    Attributes:
        path (Path): Directory path where files will be downloaded.
        version (str): Dataset version to download.
        chunk_size (int): Size of chunks for streaming download in bytes.
    """

    path: Path
    version: str
    chunk_size: int

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that the path exists and is a directory.

        Args:
            v (Path): Path to validate.

        Returns:
            Path: The validated path.

        Raises:
            RuntimeError: If path doesn't exist or is not a directory.
        """
        if not v.exists():
            raise RuntimeError("Provided path must exist.")
        if not v.is_dir():
            raise RuntimeError("Provided path must be a directory.")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that the version is supported.

        Args:
            v (str): Version string to validate.

        Returns:
            str: The validated version string.

        Raises:
            ValueError: If version is not in FIGSHARE_VERSION_IDS.
        """
        if v not in FIGSHARE_VERSION_IDS:
            raise ValueError(f"Unknown dataset version: {v}.")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate that chunk size is positive.

        Args:
            v (int): Chunk size to validate.

        Returns:
            int: The validated chunk size.

        Raises:
            ValueError: If chunk size is not greater than zero.
        """
        if v <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        return v


@GeneralUtils.validate_func_args_with_pydantic(GetFigshareDataValidator)
def get_figshare_data(
    path: Path, version: str = "2.0.0", chunk_size: int = 1024 * 1024
) -> List[Path]:
    """Download requested 3W dataset version from Figshare.

    Downloads all files associated with the specified dataset version from Figshare
    to the provided directory path. Performs MD5 checksum verification to ensure
    data integrity.

    Args:
        path (Path): Target directory where files will be downloaded.
            Must exist and be a directory.
        version (str, optional): Dataset version to download. Must be one of
            the versions in FIGSHARE_VERSION_IDS. Defaults to "2.0.0".
        chunk_size (int, optional): Size in bytes for streaming download chunks.
            Defaults to 1MB (1024 * 1024).

    Returns:
        List[Path]: List of paths to the downloaded files.

    Raises:
        RuntimeError: If the target file already exists or if MD5 checksum
            verification fails.
        ValueError: If an invalid version or chunk_size is provided.

    Example:
        >>> from pathlib import Path
        >>> downloaded = get_figshare_data(Path("./data"), version="2.0.0")
        >>> print(f"Downloaded {len(downloaded)} files")

    Notes:
        - Downloads are streamed with progress bars via tqdm
        - Each file's integrity is verified using MD5 checksums
        - Will not overwrite existing files (raises RuntimeError instead)
    """
    # Fetch metadata for all files in the requested version
    known_files = requests.get(
        FIGSHARE_BASE_URL + "/articles/" + FIGSHARE_VERSION_IDS[version] + "/files"
    )
    metadata = json.loads(known_files.text)  # list of file metadata

    downloaded = []
    for meta in metadata:  # iterate through multiple files if present
        # Stream download from Figshare
        stream = requests.get(
            FIGSHARE_BASE_URL + "/file/download/" + str(meta["id"]), stream=True
        )
        stream_size = int(stream.headers.get("content-length", 0))
        hasher = hashlib.md5()
        file_path = path / meta["name"]

        # Download with progress bar
        with tqdm(
            total=stream_size, unit="B", unit_scale=True, desc=meta["name"]
        ) as pbar:
            if file_path.exists():
                raise RuntimeError(f"{str(file_path)} already exists.")
            with open(file_path, "wb") as f:
                for chunk in stream.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        pbar.update(len(chunk))

            # Verify file integrity
            if hasher.hexdigest() != meta["supplied_md5"]:
                raise RuntimeError(
                    f"Wrong checksum detected in {meta['name']}. "
                    f"Expected {meta['supplied_md5']}, got {hasher.hexdigest()}."
                )
        downloaded.append(file_path)
    return downloaded
