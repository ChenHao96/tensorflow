<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div>

# TensorFlow的安装(源码)步骤如下(v1.14.0)

## 1 由于pip安装模块是从国外访问安装速度很慢修改访问地址为国内镜像该文件
如果没有可自行创建(本文使用的系统是 Ubuntu18 LTS)
```
vim ~/.pip/pip.conf
```

添加国内镜像源(阿里云)
```
[global]
index-url=http://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host=mirrors.aliyun.com
```

## 2 安装 Python 和 TensorFlow 软件包依赖项
```
apt install python3-dev python3-pip
pip3 install --upgrade pip six numpy future wheel setuptools mock
pip3 install --upgrade keras_applications --no-deps
pip3 install --upgrade keras_preprocessing --no-deps
```

## 3 该步骤是以源码编译为例，需要拉取源码，但是国内拉去代码过慢修改host提升拉取速度
```
vim /etc/hosts
```

添加以下hosts
```
# Github
151.101.44.249 github.global.ssl.fastly.net
192.30.253.113 github.com
103.245.222.133 assets-cdn.github.com
23.235.47.133 assets-cdn.github.com
203.208.39.104 assets-cdn.github.com
204.232.175.78 documentcloud.github.com
204.232.175.94 gist.github.com
107.21.116.220 help.github.com
207.97.227.252 nodeload.github.com
199.27.76.130 raw.github.com
107.22.3.110 status.github.com
204.232.175.78 training.github.com
207.97.227.243 www.github.com
185.31.16.184 github.global.ssl.fastly.net
185.31.18.133 avatars0.githubusercontent.com
185.31.19.133 avatars1.githubusercontent.com
```
## 4 源码编译所使用的构建工具是bazel
可以使用apt安装可是tensorflow的编译版本并不支持最新版本的bazel所以这里我们下合适的版本
```
wget https://github.com/bazelbuild/bazel/releases/download/0.25.0/bazel-0.25.0-installer-linux-x86_64.sh
```

将sh文件添加可执行权限
```
sudo chmod +x bazel-0.25.0-installer-linux-x86_64.sh
```

执行bazel脚本安装bazel构建工具
```
./bazel-0.25.0-installer-linux-x86_64.sh
```

安装过后需要添加到系统环境中:

```
vim ~/.bashrc

export PATH=$PATH:$HOME/bin

source ~/.bashrc
```

## 5 拉取源码(v1.14.0)构建并编译
```
git clone https://github.com/tensorflow/tensorflow.git -b v1.14.0
cd tensorflow
./configure
```

编译tensorflow源码
以下命令限制 Bazel 的内存消耗量
```
--local_ram_resources=2048
```

对于 GCC 5 及更高版本，为了使您的编译系统与旧版 ABI 兼容，请使用
```
--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
```
以pip模式构建代码
```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

将编译后的代码转换为pip模块文件(wheel)
```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

清除构建信息
```
bazel clean
```

## 6 pip安装构建好的模块
```
pip3 install /tmp/tensorflow_pkg/tensorflow-1.14.0-${python-version}-linux_x86_64.whl
```


# License

[Apache License 2.0](LICENSE)
