from setuptools import setup
from sys import platform

## REQUIRMENTS 

# list of requirements - base
l_requirements = [
    "seaborn",
    #"pandas", #"pandas >= 0.20.0",
    "scikit-learn", #"scikit-learn >= 0.24.1",
    "missingno==0.5.2", 
    "diptest",
    "skimpy" # ,#==0.0.9", # required python >=3.8
]
# add or not according the os platform
# platform == 'win32'
# platform.startswith('linux')
# platform == 'darwin'
if platform == 'darwin' or platform.startswith('linux'):
    l_requirements.append("minepy") #  #"minepy==1.2.6",
   
        
## SETUP
    
setup(
    zip_safe = False,
    install_requires = l_requirements,
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
        ]
    },
)