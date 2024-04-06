from setuptools import setup, find_packages

setup(
    name='QuantaTools',
    version='0.1',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
