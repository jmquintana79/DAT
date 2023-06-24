from setuptools import setup

setup(
    zip_safe=False,
    install_requires=["pandas >= 0.20.0"],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
        ]
    },
)