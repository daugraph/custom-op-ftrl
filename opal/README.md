Opal gftrl op

# BUILD
## build gftrl lib
```bash
cd opal\tensorflow\optimizer\
make
```

## wrap whl package
```bash
cd ../../..
python setup.py bdist_wheel
pip install dist/tensorflow_opal-0.0.1-cp36-cp36m-linux_x86_64.whl
```

# TF 2.4.0构建

## install python3

## install bazel 3.1.0
- 下载`dist`版本
- https://docs.bazel.build/versions/master/install-compile-source.html

## install gcc 7.4.0
- https://www.jianshu.com/p/b3f96dde3f61
