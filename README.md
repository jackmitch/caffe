# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

# Windows Fork
This is not the original [Caffe Readme](https://github.com/BVLC/caffe/blob/master/README.md) but an installation guide for windows version. It is based on top of [@niuzhiheng](https://github.com/niuzhiheng/caffe) original windows port with modifications to build with the recent updates to the [main repo](https://github.com/BVLC/caffe/). It builds using MS Visual Studio 2012. There are 3 projects CaffeLib, CaffeTests, CaffeTools which build in 32, 64 bit configurations with a CPU_ONLY option also. CaffeLib builds a static library for use in other projects. CaffeTests builds the caffe test runner. CaffeTools builds the caffe command line program.

You'll have to check the git log to see where it is in relation to the main dev branch. It also includes some commits currently in pull requests regarding using OpenCV::Mat directly into FeedForward functions.

### Important note
The MS Visual Studio 2012 compiler doesn't like the way the LayerReisterer (layer_factory.hpp) is implemented. It ignores/optimises the static variables created by the REGISTER_LAYER_CREATOR out of the static library created by CaffeLib project. I tried the approipate linker settings but could fix the issue. In the end I decided to implement a caffe::InitLayerFactory function that needs to be called once at the beginning of the program, in the same way the caffe::GlobalInit function is called (see test_caffe_main.cpp). There is probably a neater way to do this as adding new layer types obviously needs an update to the InitLayerFactory function which is a pain. 

The fork should also build on non-windows environments although I've only tested it on MacOSX 10.10.

#### Want to run first before build by yourself?
Pre-compiled binaries are available [at] (https://www.dropbox.com/s/tj12e78bzb0ac82/CaffeBinariesKenRebase.7z?dl=0) 

#### Prerequisites
You may need the followings to build the code:
- Windows 64-bit
- MS Visual Studio 2012
- CUDA toolkit 6.5
- Other dependencies which you can directly download from [here](https://www.dropbox.com/s/7yxjymv21qy8e7n/3rdpartyKenRebase.7z?dl=0)

#### Build Steps
- Check out the code and switch to *windows-rebased* branch
- Download the dependency file and extract the folders inside to project root directory. It should populate the 3rdparty folder.
- Open the caffe solution file in `./build/MSVC`
- Select the build target and build (all configurations should be OK - the DebugCPUOnly builds with the CPU_ONLY flags as per main repo - i.e. no GPU support).
- Binary outputs should appear in the bin folder (debug builds are appended with a 'd' - e.g. caffed_x64.lib where release builds are caffe_x64.lib)

#### Tips
- It takes obvious longer time when you compile for the first time. Therefore please refrain from using `clean & rebuild`.
- To support different [GPU compute capabilities](http://en.wikipedia.org/wiki/CUDA#Supported_GPUs), the code is built for several compute capability versions. If you know the exact version of your GPU device, you may remove the support to other versions to speed up the compiling procedure. You may wish to take a look at #25 for more details.

#### Known Issues
- When running the tests you may need to manually delete the temp folder inside /bin to avoid "lmdb already exists" errors.

#### Bug Report
- Please create new issues in [github](https://github.com/jaleigh/caffe/issues) if you find any bug.
- If you have new pull requests, they are very welcome.
- Before you do that, you may wish to read this [wiki page](https://github.com/jaleigh/caffe/wiki) for more information.

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
