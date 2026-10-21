from glob import glob
from setuptools import setup

package_name = 'agv_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='controlheist',
    maintainer_email='ranjannilesh05@gmail.com',
    description='Ackermann AGV that follows a yellow ball and avoids static obstacles.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'base_driver = agv_follower.base_driver:main',
            'ball_detector = agv_follower.ball_detector:main',
            'follower = agv_follower.follower:main',
        ],
    },
)
