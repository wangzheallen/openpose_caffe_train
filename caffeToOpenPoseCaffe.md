# Official Caffe to OpenPose Custom Caffe



### Caffe Modification
In order to change the official Caffe to the OpenPose version:

1. Modified file(s) (search for `OpenPose` to find the editions inside each file):
    - Makefile
    - src/caffe/proto/caffe.proto
    - src/caffe/util/blocking_queue.cpp
2. New folder(s):
    - autocompile/
    - include/caffe/openpose/
    - src/caffe/openpose/
3. New file(s):
	- caffeToOpenPoseCaffe.md
4. Deleted:
	- Makefile.config.example



### Compilation
Assuming you have all the Caffe prerequisites installed, compile this custom Caffe:

```
cd autocompile/
bash compile_caffe.sh
```
