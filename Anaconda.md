# Anaconda

## 基础知识

### 环境配置

1. 凡是可执行文件，那么我们就可以在命令行直接输入它们的名称来执行，或者说，命令行中的命令本质上就是那些存放在计算机中的各种可执行文件。在win10中，这些可执行文件一般是后缀为.exe、.sys（系统文件）、.bat（批处理文件）等（这些后缀由环境变量PATHEXT指定）的文件。注意，任何命令的背后都要有一个可执行文件，而且该可执行文件要能够被系统找到（当前目录或由PATH指出）。

2. 当我们在命令行输入一个命令，比如python3，那么系统就会自动在命令行所在的目录下寻找名为python3的可执行文件，若找到则执行它，没有找到则继续在由环境变量PATH给出的目录中寻找，若找到则执行它，没有则报错。

   > 系统变量：对所有的用户起作用
   >
   > 用户变量：仅对当前用户起作用

3. 为了可以在命令行使用conda命令，有两种方法：① 在conda.exe文件所在目录打开命令行来使用conda命令；② 将conda.exe文件所在目录的路径（...\Anaconda\Scripts）添加到环境变量PATH中，然后就可以在任何目录下打开命令行来使用conda命令。

4. 同理，为了在任何地方使用python3命令，那就得把python3.exe文件所在目录的路径（...\Anaconda）添加到环境变量PATH中；为了在任何地方使用pip3命令，那就得把pip3.exe文件所在目录的路径（...\Anaconda\Scripts）添加到环境变量PATH中。

   > 解决python命令打开MS商店：在环境变量PATH中删除路径"%USERPROFILE%\AppData\Local\Microsoft\WindowsApps"，或将该路径移动到python目录路径后面。

### 环境管理

1. Anaconda在安装时会自带某一个版本的python（在安装时由用户自己选择），该版本的python就作为Anaconda基本环境中使用的python版本。

2. Anaconda可以创建其他python环境，每个环境中的python版本和包都互相独立，由各自的环境进行管理，也就是说，若环境1安装了Tensorflow，那么在环境2中就无法使用Tensorflow，除非环境2本身也安装一个Tensorflow。

3. Anaconda的环境管理是基于目录来实现的，每一个环境就对应了一个python目录，base环境对应的就是Anaconda根目录，而其余创建的环境所对应的目录都在Anaconda根目录下的envs目录中，比如有个环境叫做py32，则它对应的目录就是"...\Anaconda\envs\py32"。

4. 实现python环境隔离和切换的原理在于临时修改环境变量PATH的值。假设环境py32对应的python目录为"...\Anaconda\envs\py32"，当激活环境py32时，Anaconda就会临时在PATH变量的最前面添加路径"...\Anaconda\envs\py32"，使得输入命令python所产生的python版本为环境py32对应的python版本；当退出环境py32，重新变成基本环境时，Anaconda就会将路径"...\Anaconda\envs\py32"从PATH变量中删除，继而在它的最前面添加路径"...\Anaconda"，这时输入命令python产生的python版本就只能是基本环境所对应的python版本；当退出所有环境后，Anaconda就会把临时添加的所有路径都从PATH变量中删去，这时输入命令python就只会返回"命令找不到"的错误提示，除非我们自己手动在PATH变量中添加了某个python版本的目录。

   > where：查看命令对应的源文件位置；
   >
   > echo %PATH%：查看当前命令行的环境变量PATH。

5. 基于Anaconda在PATH变量中添加的路径，Anaconda就可以实现不同环境之间的切换和隔离。因此，Anaconda只需要记录不同环境所需要的环境路径即可，在切换不同环境时，将对应环境所需要的路径添加到变量PATH的最前面就可以实现不同环境的隔离和切换了。

6. 为了节省存储，Anaconda实际上可能不会为每一个环境都创建一个python版本。比如当两个环境的python版本都一样时，那么Anaconda只会为其中一个环境创建python版本，而对于另一个环境，Anaconda只需要将其所需python版本的路径记录为前者创建的python版本的目录就行了。

### .condarc文件

.condarc 文件记录了 Anaconda 的一些配置，这些配置会永久生效，除非 .condarc 文件被删除了。

.condarc 文件在 win10 中一般创建在 `C:\Users\username` 目录下。

1. 通过 .condarc 文件配置镜像

   先执行 `conda config --set show_channel_urls yes` 生成 .condarc 文件；

   在 .condarc 文件中添加清华的镜像通道：

   ```
   channels:
     - defaults
   show_channel_urls: true
   default_channels:
     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
   custom_channels:
     conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
     msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
     bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
     menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
     pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
     simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
   ```

   运行 `conda clean -i` 清除索引缓存，保证用的是镜像站提供的索引。

2. 启动命令行后自动激活基础环境：添加配置 `auto_activate_base: true`。

   > 也可以通过命令实现：conda config --set auto_activate_base true。
   >
   > true改为false就不会自动激活环境了。

### 在 VSCode 中使用 Jupyter

在 VSCode 的当前环境中，需要包含以下内容：

- 安装 python
- 安装 jupyter notebook：pip install notebook
- 安装 vscode 扩展：python、jupyter

## 常V用命令

2. 查看当前环境的包

   conda list

2. 查看某个指定环境的已安装包

   conda list -n python34

3. 查看某个包可以安装的版本

   conda search ipython

4. 查看某个包是否已经被安装

   pip show ipython

   conda list ipython

5. 查看某个命令的帮助

   conda install --help

7. 安装、更新、卸载包

   pip install requests

   pip install [package-name]==X.X（指定版本）

   pip install requests --upgrade

   pip uninstall requests

   或者

   conda install requests

   conda install requests=X.X

   conda update requests

   conda uninstall requests

7. 更新所有库

   conda update --all

8. 更新 conda 和 anaconda

   conda update conda

   conda update anaconda

9. 查看版本信息

   conda info

   conda info --envs（查看已安装环境）

11. 创建python版本为3.9且名字标识为exampleName的环境

    conda create -n exampleName python=3.9

12. 激活名为exampleName的环境

    conda activate exampleName

    conda activate base（进入基本环境）

13. 退出当前环境

    conda deactivate

14. 删除名为exampleName的环境

    conda remove -n exampleName --all

15. 克隆环境example2为example1

    conda create -n example1 --clone example2

16. 将anaconda切换到32位（须管理员）

    set CONDA_FORCE_32BIT=1

    （在32位anaconda中下载的python为32位的版本）

17. 将shell初始化为conda命令行环境

    conda init

    > `conda init` 的目的就是让用户无需自己配置环境，转而将一切交由Anaconda的自动化脚本来完成，使得用户直接打开命令行就可以使用Anaconda的命令。
    >
    > 比如 `conda init` 做的其中一件事情就是添加了一个注册表 `HKEY_CURRENT_USER\Software\Microsoft\Command Processor\AutoRun` 。该注册表让用户每次打开命令行时先自动运行批处理脚本"...\Anaconda\condabin\conda_hook.bat"，该脚本会自动给环境变量PATH添加执行Anaconda命令（比如conda）所需要的一些路径，这也使得我们不需要在环境变量PATH中手动配置conda命令的路径，因此，这也是我们无需手动配置环境变量就可以使用conda命令的原因。
    >
    > `conda init` 做的另一件事就是在用户每次打开命令行时自动激活环境，这个可以通过配置.condarc文件来取消。
