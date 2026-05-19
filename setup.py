from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ntb-predictor',
    version='0.1.0',
    author='Jaecyk',
    description='Machine learning models for predicting Nigerian Treasury Bills stop rates',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jaecyk/ntb-predictor',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)