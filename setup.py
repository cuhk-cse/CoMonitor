
from setuptools import setup, find_packages

setup(
    name='CoMonitor',
    description='An python package for compressive monitoring',
    long_description=open('README.md').read(),
    version='0.1dev',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    install_requires=['numpy'],
    url='https://github.com/cuhk-cse/CoMonitor',
    author='CUHK-CSE'
)
