from setuptools import setup, find_packages

setup(
    name='gym',  # Replace with your package name
    version='0.19.0',  # Replace with your package version
    description='A customized gym package',  # Replace with a short description
    long_description=open('README.md').read(),  # Optional: Include a README file
    long_description_content_type='text/markdown',  # Optional: Specify the format of the long description
    author='zhugb',  # Replace with your name
    author_email='zhugb@buaa.edu.cn',  # Replace with your email
    packages=find_packages(exclude=['tests', 'docs']),  # Automatically find package directories
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.1',
        'pyglet>=1.4.0',
        'matplotlib>=3.5.0',
        'cloudpickle>=1.2.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-forked>=1.3',
            'flake8>=3.8.4',
        ],
        'optional': [
            'atari-py==0.2.6',
            'opencv-python>=3.',
            'box2d-py~=2.3.5',
            'mujoco_py>=1.50, <2.0',
            'pygame>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
            # 'script_name=your_package.module:function',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: MIT License',  # Replace with your license if different
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Supports Python 3.6 through 3.13+
)

