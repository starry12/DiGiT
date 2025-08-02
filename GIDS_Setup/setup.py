from setuptools import find_packages
from setuptools import setup


setup(
        name = "GIDS",
        version     = '0.1.2',
        #packages=find_packages(),
        package_data={
         '':['/home/embed/Documents/GIDS/gids_module/build/BAM_Feature_Store/BAM_Feature_Store.so'] 
        }
)