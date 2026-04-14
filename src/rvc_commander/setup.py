from setuptools import setup

setup(
    name='rvc_commander',
    version='1.0.0',
    packages=[
        'rvc_commander',
        'rvc_commander.slam'
        'rvc_commander.utils'
    ],
    package_dir={'': 'src'},
)