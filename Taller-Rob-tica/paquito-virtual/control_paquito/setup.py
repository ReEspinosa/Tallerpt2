
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'control_paquito'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vero',
    maintainer_email='v.arriola@ciencias.unam.mx',
    description='Sends /cmd_vel and string commands from PS joystick',
    license='GPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ps_control_node = control_paquito.ps_control_node:main',
        ],
    },
)
