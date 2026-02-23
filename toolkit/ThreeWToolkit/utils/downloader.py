import json
import requests  # type: ignore
import hashlib

from tqdm import tqdm
from pathlib import Path

from ..utils.general_utils import GeneralUtils
from pydantic import BaseModel, field_validator

FIGSHARE_BASE_URL = "https://api.figshare.com/v2"
FIGSHARE_VERSION_IDS = {
    "1.0.0": "29205926",
    "1.1.0": "29205932",
    "1.1.1": "29205947",
    "2.0.0": "29205836",
}

# expected SHA256 digests for dataset versions
FIGSHARE_VERSION_DIGESTS = {
    "1.0.0": "34762c8d8077718f75024398996bbc57669f72234084b750f0bcaecfb7e85402",
    "1.1.0": "a4d99b1783333f4ca7389e45e5bab0188aeb5c381a6439c8a3f634bd96ab9d82",
    "1.1.1": "fed2af7c6f607b46d4963fe7eb7ee7ada7df3cfde0885f205a6408d0c2adff69",
    "2.0.0": "c037a6a9d5dcc9b2add7b9ea413f5cff367c57bad3a93faac1d1d32da27fcbea",
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
    def validate_path(cls: type["GetFigshareDataValidator"], path: Path) -> Path:
        """Validate that the path exists and is a directory.

        Args:
            cls (GetFigshareDataValidator): The class reference.
            path (Path): Path to validate.

        Returns:
            Path: The validated path.

        Raises:
            RuntimeError: If path doesn't exist or is not a directory.
        """
        if not path.exists():
            raise RuntimeError("Provided path must exist.")
        if not path.is_dir():
            raise RuntimeError("Provided path must be a directory.")
        return path

    @field_validator("version")
    @classmethod
    def validate_version(cls: type["GetFigshareDataValidator"], version: str) -> str:
        """Validate that the version is supported.

        Args:
            cls (GetFigshareDataValidator): The class reference.
            version (str): Version string to validate.

        Returns:
            str: The validated version string.

        Raises:
            ValueError: If version is not in FIGSHARE_VERSION_IDS.
        """
        if version not in FIGSHARE_VERSION_IDS:
            raise ValueError(f"Unknown dataset version: {version}.")
        return version

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(
        cls: type["GetFigshareDataValidator"], chunk_size: int
    ) -> int:
        """Validate that chunk size is positive.

        Args:
            cls (GetFigshareDataValidator): The class reference.
            chunk_size (int): Chunk size to validate.

        Returns:
            int: The validated chunk size.

        Raises:
            ValueError: If chunk size is not greater than zero.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        return chunk_size


@GeneralUtils.validate_func_args_with_pydantic(GetFigshareDataValidator)
def get_figshare_data(
    path: Path, version: str = "2.0.0", chunk_size: int = 1024 * 1024
) -> list[Path]:
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
        list[Path]: List of paths to the downloaded files.

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
        stream = requests.get(meta["download_url"], stream=True)
        stream_size = int(stream.headers.get("content-length", 0))
        hasher = hashlib.sha256()
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
            if hasher.hexdigest() != FIGSHARE_VERSION_DIGESTS[version]:
                raise RuntimeError(
                    f"Wrong checksum detected in {meta['name']}. "
                    f"Expected {FIGSHARE_VERSION_DIGESTS[version]}, got {hasher.hexdigest()}."
                )
        downloaded.append(file_path)
    return downloaded
