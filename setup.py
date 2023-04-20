from setuptools import find_packages, setup


HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str):
    requirements = []
    with open(file_path, 'rb') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='DeliveryTimePredictionProject',
    version='0.0.1',
    author='Abhijit Chakraborty',
    author_email='abhijityachak@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
