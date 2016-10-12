from __future__ import absolute_import
from setuptools import setup, find_packages
import os

os.chdir(u'lib')
os.system(u'make CFLAGS=-O3')
os.chdir(u'../')

setup(name = u'xraycam',
    packages = find_packages(u'.'),
    package_dir = {u'xraycam': u'xraycam'},
    package_data = {u'xraycam': [u'data/*', u'../lib/*']},
#    scripts = [
#        'bin/oacapture'
#        ],
    install_requires = [u'paramiko', u'numpy', u'matplotlib', u'mpld3', u'plotly', u'humanfriendly', u'multiprocess'],
    zip_safe = False)

# TODO: the below config isn't getting loaded (instead python grabs the package version). Fix this.
import shutil
configfile = os.path.expanduser(u'~/.xraycam/detconfig.py')
configdir = os.path.dirname(configfile)
if not os.path.exists(configdir):
    os.makedirs(configdir)
shutil.copyfile(u'xraycam/detconfig.py', configfile)

