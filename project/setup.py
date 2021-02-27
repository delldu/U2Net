"""Setup."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 02月 27日 星期六 17:25:55 CST
# ***
# ************************************************************************************/
#

import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
     name='matting',  
     version='0.0.1',
     author='Dell',
     author_email="",
     description="matting",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url='https://github.com/dell/matting',
     packages=['matting'],
     package_data={'matting': ['weights/*.pth']},
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )
