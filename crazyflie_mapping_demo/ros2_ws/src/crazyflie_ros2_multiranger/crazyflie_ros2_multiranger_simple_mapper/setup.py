from setuptools import find_packages, setup

package_name = 'crazyflie_ros2_multiranger_simple_mapper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kimberly McGuire',
    maintainer_email='kimberly@bitcraze.io',
    description='Simple mapper for Crazyflie using odometry and multiranger data',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_mapper_multiranger = crazyflie_ros2_multiranger_simple_mapper.simple_mapper_multiranger:main',
        ],
    },
)
