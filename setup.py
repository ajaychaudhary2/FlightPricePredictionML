from setuptools import find_packages,setup
from typing import List

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""

setup(
    name='FlightPricePRediction',               # Name of the package
    version='0.0.1',                            # Version of the package
    author='Ajay Chaudhary',                    # Your name
    author_email='ajaych2822@gmail.com',        # Your email
    description='A package for flight price prediction',  # Short description
    install_requires=["scikit-learn", "pandas", "numpy"],  # Required dependencies
    packages=find_packages(where='src'),        # Finds all packages in the `src` directory
    package_dir={'': 'src'},                    # Sets the package root directory as `src`
    python_requires='>=3.6',                    # Ensures compatibility with Python 3.6+
)