import glob
import shutil
from distutils.util import convert_path

from setuptools import find_packages, setup, Command
import os
import io

# package metadata
NAME = "bee_classifier"

DESCRIPTION = 'python package that trains an image classifier to predict if a bee in carrying pollen'
EMAIL = 'mgmoesta@gmail.com'
AUTHOR = 'MGM'
REQUIRES_PYTHON = '>=3.7.0'



# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
main_ns={}
ver_path = convert_path('python/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# # Load the package's __version__.py module as a dictionary.
# about = {}
# if not VERSION:
#     project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
#     with open(os.path.join(here, project_slug, '__version__.py')) as f:
#         exec(f.read(), about)
# else:
#     about['__version__'] = VERSION


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)


# Where the magic happens:
setup(
    name=NAME,
    version=main_ns['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    dependency_links=[
    ],
    # url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    # package_data={'': ['defaults.yml']},
    #     # install_requires=REQUIRED,
    #     # extras_require=EXTRAS,
    #     # include_package_data=True,
    #     # classifiers=[
    #     #     # Trove classifiers
    #     #     # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    #     #     'License :: OSI Approved :: MIT License',
    #     #     'Programming Language :: Python',
    #     #     'Programming Language :: Python :: 3',
    #     #     'Programming Language :: Python :: 3.6',
    #     #     'Programming Language :: Python :: Implementation :: CPython',
    #     #     'Programming Language :: Python :: Implementation :: PyPy'
    #     # ],
    # $ setup.py publish support.
    cmdclass={
        'clean': CleanCommand
    },
)