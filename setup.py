from setuptools import setup, find_packages

setup(
    name="blur_weather",
    version="0.1.0",
    packages=find_packages(),
    package_data={"blur_weather": ["reference/*"]},
)
