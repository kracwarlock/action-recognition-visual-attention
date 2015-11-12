### Data Format
The `train_features.h5` file is an HDF5 file with `DATATYPE H5T_IEEE_F32LE` and its `DATASPACE` is `SIMPLE { ( #frames, 7*7*1024 ) / ( H5S_UNLIMITED, H5S_UNLIMITED ) }` and `DATASET "features"`.

The `train_framenum.txt` file contains #frames for each video:
```
89
123
22
136
```

The `train_filename.txt` file contains the video filenames relative to the root video directory:
```
cartwheel/lea_kann_radschlag_cartwheel_f_cm_np1_ri_med_0.avi
cartwheel/park_cartwheel_f_cm_np1_ba_med_0.avi
catch/96-_Torwarttraining_1_catch_f_cm_np1_le_bad_0.avi
catch/Ball_hochwerfen_-_Rolle_-_Ball_fangen_(Timo_3)_catch_f_cm_np1_le_goo_0.avi
```

The `train_labels.txt`file for uni-label datasets looks like
```
0
7
43
```
and for multi-label datasets:
```
0,0,0,0,0,0,0,1,0,0,0,0
0,0,0,0,0,0,0,1,0,0,0,0
0,0,0,0,0,0,1,1,0,0,0,0
0,0,0,0,0,0,0,0,0,0,0,1
```
The same format is required for the validation and test files too.

### data_handler.py
We have used `order='F'` in all our `numpy.reshape()` calls since we created our data file using Matlab which uses the Fortran indexing order.
You will have to remove this parameter if that is not the case with you.

### GPU locking
Toronto ML users need not make any modifications. The script locks a free GPU automatically.
Non-Toronto users can adapt the GPU locking scripts or remove the following lines from the `scripts/evaluate_*` files:
```
import util.gpu_util
board = util.gpu_util.LockGPU()
print 'GPU Lock Acquired'

util.gpu_util.FreeGPU(board)
print 'GPU freed'
```
