from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="spectrum_analyzer_tool",
    version="1.0.0",
    description="Analyze spectrum analyzer recordings for signal characteristics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anthonydesantiago.github.io/Spectrum_Analyzer_Tool/",
    author="Anthony DeSantiago, Camille Reaves, Geonhyeong Kim, Jalon Bailey, Matthew Morgan",
    author_email="",
    packages=["spectrum_analyzer_tool"],
    include_package_data=True,
    install_requires=[
        "decord", "easyocr", "matplotlib", "numpy", "opencv_python", "pandas", "PyQt6", "PyQt6_sip", "torch", "ultralytics"
    ],
    entry_points={"console_scripts": ["spectrum_analyzer_tool=spectrum_analyzer_tool.__main__:main"]},
)
