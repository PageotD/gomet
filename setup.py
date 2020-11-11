from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='gomet',
    version='0.1.0',
    description='GOMet package',
    long_description=readme,
    author='Damien Pageot',
    author_email='damien.pageot@gmail.com',
    url='https://github.com/PageotD/gomet',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
