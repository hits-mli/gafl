from setuptools import setup

setup(
    name="gafl",
    packages=[
        'gafl',
        'openfold'
    ],
    package_dir={
        'gafl': './gafl',
        'openfold': './openfold',
    },
)
