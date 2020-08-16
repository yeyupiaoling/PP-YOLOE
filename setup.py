import setuptools

install_requires = ['tqdm', 'typeguard', 'visualdl>=2.0.0b', 'opencv-python', 'PyYAML', 'shapely']

setuptools.setup(name='ppdet',
                 version='0.0.4',
                 description='PaddlePaddle PaddleDetection',
                 url='https://github.com/PaddlePaddle/PaddleDetection',
                 author='yeyupiaoling',
                 author_email='yeyupiaoling@foxmail.com',
                 license='Apache License 2.0',
                 packages=setuptools.find_packages(),
                 install_requires=install_requires,
                 zip_safe=False)
