#setup.py
from setuptools import setup, find_packages

setup(
    name='ZCPstatespy',
    version='0.1',
    packages=find_packages(),
    description='A custom machine learning visualization toolkit',
    author='Chuanping Zhao',
    author_email='zcphbumaster@163.com',
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'seaborn',
        'scikit-learn',
        'scipy'
    ],
    include_package_data=True,
)