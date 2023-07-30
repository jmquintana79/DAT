from setuptools import setup

setup(
    zip_safe=False,
    install_requires=[
        "pandas >= 0.20.0",
        "scikit-learn >= 0.24.1",
        "missingno==0.5.1",
        #"matplotlib==3.5.1",
        "skimpy==0.0.9", # required python >=3.8 
        "minepy==1.2.6",
        "unidip==0.1.1"
        ],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
        ]
    },
)