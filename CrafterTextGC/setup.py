from setuptools import setup, find_packages

setup(
    name='CrafterTextGC',
    version='0.1.0',
    description='A text-based version of the Crafter environment which is goal-conditioned.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Loris Gaven',
    url='https://github.com/lorisgaven/CrafterTextGC',
    packages=find_packages('.'),
    include_package_data=True,
    package_data={
        'crafter_text_gc.crafter': ['data.yaml'],
    },
    install_requires=[
        'gym==0.26.2',
        'ruamel.yaml==0.17.21',
        'ruamel.yaml.clib==0.2.7',
        'imageio==2.22.4',
        'imageio-ffmpeg==0.4.7',
        'opensimplex==0.4.3'
    ]
)
