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
    path: Path
    version: str
    chunk_size: int

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        if not v.exists():
            raise RuntimeError("Provided path must exist.")
        if not v.is_dir():
            raise RuntimeError("Provided path must be a directory.")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v not in FIGSHARE_VERSION_IDS:
            raise ValueError(f"Unknown dataset version, {v}.")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        return v


@GeneralUtils.validate_func_args_with_pydantic(GetFigshareDataValidator)
def get_figshare_data(
    path: Path, version: str = "2.0.0", chunk_size: int = 1024 * 1024
) -> List[Path]:
    """
    Download requested 3W version from figshare into 'path'.
    """
    known_files = requests.get(
        FIGSHARE_BASE_URL + "/articles/" + FIGSHARE_VERSION_IDS[version] + "/files"
    )
    metadata = json.loads(known_files.text)  # list

    downloaded = []
    for meta in metadata:  # maybe multiple files
        stream = requests.get(
            FIGSHARE_BASE_URL + "/file/download/" + str(meta["id"]), stream=True
        )
        stream_size = int(stream.headers.get("content-length", 0))
        hasher = hashlib.md5()
        file_path = path / meta["name"]

        with tqdm(
            total=stream_size, unit="B", unit_scale=True, desc=meta["name"]
        ) as pbar:
            if file_path.exists():
                raise RuntimeError(f"{str(file_path)} already exists.")
            with open(file_path, "wb") as f:
                for chunk in stream.iter_content(
                    chunk_size=chunk_size
                ):  # write in chunks
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        pbar.update(len(chunk))

            if hasher.hexdigest() != meta["supplied_md5"]:
                raise RuntimeError(
                    f"Wrong checksum detected in {meta['name']}. Expected {meta['supplied_md5']}, got\
                                   {hasher.hexdigest()}."
                )
        downloaded.append(file_path)
    return downloaded
