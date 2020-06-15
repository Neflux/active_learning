from setuptools import setup
from glob import glob
import os

package_name = 'elohim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
                   ('share/ament_index/resource_index/packages',
                    ['resource/' + package_name]),
                   ('share/' + package_name, ['package.xml'])]
               + [("share/" + package_name + "/" + d, [os.path.join(d, f) for f in files]) for d, _, files in
                  os.walk("models")]
    + [('share/' + package_name + '/worlds', glob('worlds/*.world'))]
    , install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stefano Bonato',
    maintainer_email='bonats@usi.ch',
    description='World editor',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'init = elohim.world_init:main',
            'start = elohim.random_controller:main',
            'record = elohim.historia:main',
        ],
    },
)
