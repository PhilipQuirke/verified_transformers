from setuptools import setup, find_packages

setup(
    name='QuantaTools',
    version='1.1',
    packages=find_packages(),
    description='Tools for use with the quanta framework annd transformer models',
    author='Philip Quirke',
    author_email='philipquirkenzgmail.com',
    install_requires=[
        'numpy>=1.18.1',
        'wheel'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
