from setuptools import setup, find_packages

setup(name = 'xraycam',
    packages = find_packages('.'),
    package_dir = {'xraycam': 'xraycam'},
    package_data = {'xraycam': ['data/*']},
    scripts = [
        'bin/oacapture'
        ],
    install_requires = ['paramiko', 'numpy', 'matplotlib', 'mpld3', 'plotly', 'humanfriendly'],
    zip_safe = False)
