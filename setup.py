from setuptools import setup, find_packages
import os

os.chdir('lib')
os.system('make CFLAGS=-O3')
os.chdir('../')

setup(name = 'xraycam',
    packages = find_packages('.'),
    package_dir = {'xraycam': 'xraycam'},
    package_data = {'xraycam': ['data/*', '../lib/*']},
#    scripts = [
#        'bin/oacapture'
#        ],
    install_requires = ['paramiko', 'numpy', 'matplotlib', 'mpld3', 'plotly', 'humanfriendly', 'multiprocess'],
    zip_safe = False)
