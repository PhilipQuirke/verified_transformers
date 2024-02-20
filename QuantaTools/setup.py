from setuptools import setup, find_packages

setup(
    name='QuantaTools',
    version='0.1',
    packages=['QuantaTools']
    description='Tools for use with the quanta framework annd transformer models',
    author='Philip Quirke',
    author_email='philipquirkenzgmail.com',
    url='https://github.com/PhilipQuirke/verified_transformers/QuantaTools',
    install_requires=[
        'numpy>=1.18.1',  # Example dependency, specify your package's dependencies here
        # Add other dependencies as needed
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
