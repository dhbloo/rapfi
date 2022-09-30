# Rapfi

## Overview

Rapfi is a powerful Gomoku/Renju (Five in a row) playing engine supporting [Piskvork protocol](http://petr.lastovicka.sweb.cz/protocl2en.htm). Since Piskvork protocol requires AI to communicate through standard input and output, in order to use it comfortably, you may need the compatible GUI such as [Piskvork](https://github.com/wind23/piskvork_renju) or [Yixin-Board](https://github.com/dhbloo/Yixin-Board). Piskvork interface can be  downloaded from [here](https://raw.githubusercontent.com/wind23/piskvork_renju/master/Release/piskvork_renju.zip).

Rapfi engine currently uses alpha-beta search algorithm with classical and NNUE (Efficiently Updateable Neural Network) evaluation. NNUE evaluations achieves more precise evaluation with a significantly larger computation which can benefit from the vector instructions (such as SSE or AVX).

### Subprojects

This repository contains the main Rapfi engine project and several releated subprojects:

+ Rapfi - Gomoku/Renju AI engine
+ Gomocalc - Gomoku/Renju web application
+ Trainer - NNUE evaluator training codes
+ Networks - A submodule that holds the latest network weights

## Usage

To use Rapfi engine, first acquire the engine executable either by compiling it yourself, or getting it from Github releases. After that, acquire both classical and NNUE evaluation weights from [the network repo](https://github.com/dhbloo/rapfi-networks). The config file is also available in the repo. Place all weight files and config file in the directory containing Rapfi engine, then Rapfi will automatically find and load required files.

### Build Rapfi from source

You can build the engine from source on an environment with [CMake](https://cmake.org/) installed. Rapfi supports compilation with common compilers with C++17 supports such as msvc, gcc and clang. With cmake and compiler installed and available from shell, Rapfi can be compiled with following commands.

```bash
cd Rapfi
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

The default setup builds Rapfi with multi-threading and AVX2 support. To build it for running on older CPU without AVX2 support, AVX2 and AVX usage can be disable through CMake options. To see description for other CMake compile options, please refer to `CMakeLists.txt`.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_AVX=OFF -DUSE_AVX2=OFF
```

### Supported Piskvork protocol extensions

(To be added)



