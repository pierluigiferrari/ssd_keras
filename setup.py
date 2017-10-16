from setuptools import setup, find_packages

setup(
    name='singleshot',
    version='2017.10.16.0',
    description='A simple SingleShotDetector package',
    url='https://github.com/lababidi/singleshot',
    author='Mahmoud Lababidi',
    author_email='mla@mla.im',
    license='GPL',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['keras','sh', 'regex', 'numpy', 'pandas', 'rasterio'],
    python_requires='>=3',
    entry_points={
        'console_scripts': [
            'trainssd=trainssd:console',
        ],
    },
)
