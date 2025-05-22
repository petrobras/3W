from setuptools import setup, find_packages

setup(
    name="3WToolkit",
    version="0.1.0",
    author="3WToolkit Team",
    author_email="matheus.cn10@gmail.com",
    description="",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mathtzt/3WToolkit",
    packages=find_packages(exclude=["tests*", "examples*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
            "jupyter"
        ],
        "docs": [
            "mkdocs",
            "mkdocstrings[python]",
        ]
    },
    include_package_data=True,
    license="GPL-3.0",
)