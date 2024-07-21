

import setuptools


with open('README.md', 'r', encoding = 'utf-8') as f:
    long_description = f.read()



__version__ = "0.0.0.0"

REPO_NAME = 'waferDetection'
AUTHOR_USER_NAME = 'MadanuAugustin'
SRC_REPO = 'waferDetection'
AUTHOR_EMAIL = 'augustin7766@gmail.com'


setuptools.setup(
    name = SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description= 'A small python package for CNN app',
    long_description=long_description,
    url = "https://github.com/MadanuAugustin/Wafer_defect_classification_using_transfer_learning.git",
    package_dir={"" : "src"},
    packages=setuptools.find_packages(where="src")
)