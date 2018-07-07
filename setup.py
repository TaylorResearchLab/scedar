from setuptools import setup, find_packages


setup(
    name='scedar',
    version='0.0.1.dev1',
    description='Single-cell explorative data analysis for RNA-Seq',
    url='http://github.com/logstar/scedar',
    author='Yuanchao Zhang',
    author_email='logstarx@gmail.com',
    license='MIT',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 1 - Planning',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='~=3.5',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'sklearn',
        'xgboost',
        'networkx',
        'fa2',
        'umap-learn'
    ],
    setup_requires=[
        'pytest-mpl',
        'pytest-xdist',
        'pytest-cov',
        'pytest-runner',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'sklearn',
        'xgboost',
        'networkx',
        'fa2',
        'umap-learn'
    ],
    tests_require=[
        'pytest'
    ],
)
