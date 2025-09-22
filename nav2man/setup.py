from setuptools import setup, find_packages

setup(
    name="nav2man",
    packages=[
        package for package in find_packages() if package.startswith("nav2man")
    ],
    install_requires=[],
    version="0.0.1",
)