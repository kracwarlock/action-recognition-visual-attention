### Running the code
In the action-recognition-visual-attention directory you can use the following commands to run the script files:
```
THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN,nvcc.fastmath=True' python -m scripts.evaluate_ucf11
THEANO_FLAGS='floatX=float32,device=gpu1,mode=FAST_RUN,nvcc.fastmath=True' python -m scripts.evaluate_mAP
THEANO_FLAGS='floatX=float32,device=gpu2,mode=FAST_RUN,nvcc.fastmath=True' python -m scripts.evaluate_hmdb51gln
```

### Visualizations
The file `draw-visualizations.ipynb` is a sample IPython notebook for drawing visualizations.
