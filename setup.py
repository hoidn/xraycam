from setuptools import setup, find_packages

setup(name = 'xraycam',
    packages = find_packages('.'),
    package_dir = {'xraycam': 'xraycam'},
    package_data = {'xraycam': ['data/*']},
    install_requires = ['paramiko', 'numpy', 'matplotlib'],
    zip_safe = False)
