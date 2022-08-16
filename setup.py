"""
SpatialDM: Spatial co-expression Detected by bivariate Moran
SpatialDM: Spatial ligand-receptor co-expression for Direct Messaging
See: https://github.com/leeyoyohku/SpatialDM
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Set __version__ for the project.
exec(open("./spatialdm/version.py").read())

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    
reqs = [
    'numpy>=1.9.0', 
    'scipy>=1.4.0', 
    'matplotlib', 
    'statsmodels>=0.11.1',
    'scanpy',
    'pandas',
    'sklearn',
    'seaborn'
    ]

setup(
    name='SpatialDM',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='SpatialDM: Spatial co-expression Detected by bivariate Moran',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/leeyoyohku/SpatialDM',

    # Author details
    author=['Zhouxuan Li', 'Tianjie Wang', 'Yuanhua Huang'],
    author_email='leeyoyo@connect.hku.hk',

    # Choose your license
    license='Apache-2.0',

    # What does your project relate to?
    keywords=[
        'Spatial transcriptomics', 
        'Spatial association', 
        'Ligand-recptor interaction'
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # entry_points={
    #       'console_scripts': [
    #           ],
    #       }, 

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    
    install_requires=reqs,

    extras_require={
        'docs': [
            #'sphinx == 1.8.3',
            'sphinx_bootstrap_theme']},

    py_modules = ['SpatialDM']

    # buid the distribution: python setup.py sdist
    # upload to pypi: twine upload dist/...
)
