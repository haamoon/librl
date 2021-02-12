# librl #

### Installation ###
Tested in Ubuntu 16.04 and Ubuntu 18.04 with python 3.5, 3.6, 3.7.

#### Install librl and most of the requirements ####
Install this repo and requirements:
```
git clone https://github.com/cacgt/librl.git
pip install --upgrade -r requirements.txt
```
You may need to run
```
export PYTHONPATH="{PYTHONPATH}:[the parent folder of librl repo]"
```
The current version requires also tensorflow2.
```
pip install --upgrade tensorflow
```

#### Install DART ####
The Ubuntu package is too new for PyDart2, so we install it manually. 

First install the requirements following the instructions of Install DART from source at https://dartsim.github.io/install_dart_on_ubuntu.html. 
Next we compile and install DART manually, because PyDart2 only supports DART before 6.8.
```
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.7.2
mkdir build
cd build
cmake ..
make -j4
sudo make install
```
Someitmes the library may need to be linked manually.
```
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib" >> ~/.bashrc
```

#### Install PyDart2 ####
Installing PyDart2 through pip does not work, so we install it manually.
```
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python setup.py build build_ext
python setup.py develop
```


#### Install DartEnv ####
This is a slightly modified version of [DartEnv](https://github.com/DartEnv/dart-env). The changes include:

* Make nodisplay as default.
* Add a state property for convenience.

To install it, 
```
git clone https://github.com/gtrll/dartenv.git
cd dartenv
git checkout nodisplay
pip install -e .[dart]
```

#### Run experiments ####
Firstly, go to the main folder.
```
cd librl
```
Run a single script, e.g., Policy Gradient in `scripts/pg.py`
```
python scripts/pg.py
```
Run `scripts/pg.py` with a set of different configurations given as a dict named `range_common` in `scripts/pg_ranges.py`
```
python batch_run.py pg -r common
```
