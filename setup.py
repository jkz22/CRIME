from setuptools import setup, find_packages

setup(
    name="crime",
    version="0.1",
    description='A package for the CRIME project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jihan Zaki',
    author_email='jihan.zaki@outlook.com',
    url='https://github.com/jkz22/CRIME',
    packages=find_packages(),
    license='LICENSE',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'lime',
        'matplotlib'
    ],
    python_requires='>=3.6',
)
