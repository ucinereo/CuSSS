from pathlib import Path
import subprocess
import sys
import os

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

from wheel.bdist_wheel import bdist_wheel

class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]

class CMakeBuild(build_ext):
    """
    Custom build_text command that builds the extensions using CMake.
    """

    def run(self):
        """Setup cmake build.
        """
        root = Path(__file__).resolve().parent
        source_dir = root / "cusss"
        build_dir = root / "build" / "cmake-build"
        install_dir = Path(self.build_lib).resolve() / "cusss"

        build_dir.mkdir(parents=True, exist_ok=True)
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home is None:
            sys.exit("$CUDA_HOME undefined.")
        
        self._run_cmake(source_dir, build_dir, install_dir)

    
    def _run_cmake(self, source_dir: Path, build_dir: Path, install_dir: Path) -> None:
        """Configure cmake parameters and start compilation defined in cusss/CMakeLists.txt.
        """

        print(f"Installing into: {install_dir}")
        # Configure cmake options
        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]

        print(f"CMake configuration options: {cmake_options}")
        subprocess.run(
            ["cmake", source_dir, *cmake_options], cwd=build_dir, check=True,
        )

        print("Start CMake build process...")
        subprocess.run(
            ["cmake", "--build", build_dir, "--target", "install", "-j", "8"],
            check=True,
        )

if __name__ == "__main__":
    setup(
        name="cusss",
        version="0.1",
        description="CUDA implementation of SSS variants",
        ext_modules=[Extension(name="cusss", sources=[])],
        cmdclass={
            "build_ext": CMakeBuild,
            "bdist_wheel": universal_wheel,
        },
        package_data={"cusss": ["cusss/lib/*", "cusss/include/*"]}
    )
