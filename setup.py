from setuptools import setup, find_packages
print(find_packages())
required_packages = []
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
      name='aracle',
      version='0.1.0',
      author='Ji Won Park',
      author_email='jiwon.christine.park@gmail.com',
      packages=find_packages(),
      license='LICENSE.md',
      description='Solar active region prediction using deep learning',
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/jiwoncpark/aracle',
      install_requires=required_packages,
      #dependency_links=[],
      include_package_data=True,
      #entry_points={
      #'console_scripts': ['generate=baobab.generate:main',],
      #},
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['Development Status :: 4 - Beta',
      'License :: OSI Approved :: BSD License',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python'],
      keywords='physics'
      )