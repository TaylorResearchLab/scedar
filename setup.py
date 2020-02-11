from setuptools import setup, find_packages

with open("pypiREADME.md", "r") as fh:
    long_description = fh.read()

setup(
    name='scedar',
    version='0.2.0',
    url='http://github.com/logstar/scedar',
    author='Yuanchao Zhang',
    author_email='logstarx@gmail.com',
    description='Single-cell exploratory data analysis for RNA-Seq',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='>=3.5, <3.8',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib<=3.1.0',
        'pandas',
        'seaborn<=0.9.0',
        'scikit-learn<=0.21',
        'xgboost',
        'networkx',
        'fa2',
        'umap-learn',
        'python-igraph',
        'nmslib',
        'leidenalg'
    ],
    setup_requires=[
        'coverage',
        'pytest-runner',
        'pytest-cov',
        'pytest-mpl',
        'numpy',
        'scipy',
        'matplotlib<=3.1.0',
        'pandas',
        'seaborn<=0.9.0',
        'scikit-learn<=0.21',
        'xgboost',
        'networkx',
        'fa2',
        'umap-learn',
        'python-igraph',
        'nmslib',
        'leidenalg'
    ],
    tests_require=[
        'pytest'
    ],
)
