from setuptools import setup, find_packages
print(find_packages('.'))

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
      #dependency_links=[],
      include_package_data=True,
      entry_points={
      'console_scripts': [
      'generate_toy_data=aracle.toy_data.generate_toy_data:main',
      ],
      },
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