from setuptools import setup, find_packages

setup(
    name="n2m",
    packages=[
        package for package in find_packages() if package.startswith("n2m")
    ],
    install_requires=[],
    version="0.0.1",
)