"""Setup configuration for ScenarioKnowledgeGraph."""

from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import sys
import subprocess


class CMakeBuild(build_ext):
    """Build C++ extension using CMake."""

    def run(self) -> None:
        try:
            subprocess.check_call(["cmake", "--version"])
        except (OSError, FileNotFoundError):
            print("CMake not found; skipping C++ extension build")
            return

        import pybind11

        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={Path(self.build_lib).absolute()}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ]

        try:
            subprocess.check_call(["cmake", "-S", "cpp", "-B", str(build_dir)] + cmake_args)
            subprocess.check_call(["cmake", "--build", str(build_dir), "-j", "4"])
            print("C++ extension built successfully")
        except subprocess.CalledProcessError:
            print("C++ extension build failed; falling back to NumPy")


setup(
    name="skg",
    version="0.1.0",
    description="Scenario Knowledge Graph",
    packages=find_packages(),
    ext_modules=[],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.11",
)
