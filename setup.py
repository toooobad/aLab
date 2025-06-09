# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from setuptools import find_packages, setup


"""
python setup.py命令是用于构建、安装和分发Python包的命令。它需要在Python包的根目录下运行。

常用的一些参数和说明如下:

    build: 构建包,生成构建文件
    install: 安装包
    sdist: 创建源代码分发文件
    bdist: 创建二进制分发文件
    clean: 清除构建文件和缓存文件
    develop: 安装包并支持开发模式,即在安装后仍可编辑源代码
    --user: 将包安装到当前用户目录下

    例如:
        运行以下命令可以构建并安装一个Python包:

        python setup.py build
        python setup.py install

"""

version_file = 'aLab/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    setup(
        name='aLab',
        version=get_version(),
        author='ace',
        description="Ace's Deep Learning Algorithms Library",
        long_description=readme(),
        author_email='aixinjin@outlook.com',
        keywords='deep learning algorithms',
        url='https://gitee.com/toooBad/aLab',
        packages=find_packages(exclude=('configs')),
        license='MIT',
        )