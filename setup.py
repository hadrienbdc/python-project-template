from setuptools import setup, find_packages

setup(
    name="Simple Python project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'joblib',
        'sklearn',
        'pandas'
    ],
    tests_require=[
        'unittest2',
        'pyhamcrest',
        'pytest'
    ],
    setup_requires=[
        'pytest-runner'
    ]
)
