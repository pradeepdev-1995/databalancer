import setuptools


with open('README.md') as f:
    README                                = f.read()

setuptools.setup(
    author                                ="Pradeep T",
    author_email                          ='pradeeprajkvr@gmail.com',
    name                                  ='databalancer',
    license                               ='Apache License 2.0',
    description                           ='Databalancer is the python library dedicated to balance the imbalanced text classification datasets before the model training in machine learning applications',
    version                               ='v0.0.9',
    long_description                      =README,
    long_description_content_type         ='text/markdown',
    url                                   ='https://github.com/pradeepdev-1995/databalancer',
    packages                              =['databalancer'],
    python_requires                       =">=3.6.9",
    install_requires                      =[
                                                'numpy==1.17.4',
                                                'pandas==1.1.4',
                                                'pytorch-lightning==0.7.5',
                                                'pytorch-transformers==1.1.0',
                                                'torch==1.8.0',
                                                'torchfile==0.1.0',
                                                'pytorch-lightning==0.7.5',
                                                'transformers==4.9.1',
                                                'matplotlib==3.3.3',
                                                'PyQt5==5.15.7',
                                                'PyQt5-Qt5==5.15.2',
                                                'PyQt5-sip==12.11.0',
                                                'pyRFC3339==1.0',
                                                'nlpaug==1.1.10',
                                                'textattack==0.3.5',
                                                'fastt5==0.1.4'

                                            ],
    classifiers                           = [
                                            'Development Status :: 5 - Production/Stable',
                                            'License :: OSI Approved :: Apache Software License',
                                            'Programming Language :: Python :: 3.6',
                                            'Topic :: Software Development :: Libraries',
                                            'Topic :: Software Development :: Libraries :: Python Modules',
                                            'Intended Audience :: Developers'
                                              ]
                                          )
