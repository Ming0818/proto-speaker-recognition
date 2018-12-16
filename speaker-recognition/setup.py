from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='helloworld',
    version='0.1.0',
    description='Package for Hello World project',
    long_description=readme,
    author='Jack Sparrow',
    author_email='sparrow@example.com',
    url='https://github.com/mediadataes/hello_world_proto',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
