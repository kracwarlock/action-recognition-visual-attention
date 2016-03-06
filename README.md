## Action Recognition using Visual Attention

We propose a soft attention based model for the task of action recognition in videos. 
We use multi-layered Recurrent Neural Networks (RNNs) with Long-Short Term Memory 
(LSTM) units which are deep both spatially and temporally. Our model learns to focus 
selectively on parts of the video frames and classifies videos after taking a few 
glimpses. The model essentially learns which parts in the frames are relevant for the 
task at hand and attaches higher importance to them. We evaluate the model on UCF-11 
(YouTube Action), HMDB-51 and Hollywood2 datasets and analyze how the model focuses its 
attention depending on the scene and the action being performed.

## Dependencies

* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [Theano](http://www.deeplearning.net/software/theano/)
* [h5py](http://docs.h5py.org/en/latest/)

## Reference

If you use this code as part of any published research, please acknowledge the
following papers:

**"Action Recognition using Visual Attention."**  
Shikhar Sharma, Ryan Kiros, Ruslan Salakhutdinov. *[arXiv](http://arxiv.org/abs/1511.04119)*

    @article{sharma2015attention,
        title={Action Recognition using Visual Attention},
        author={Sharma, Shikhar and Kiros, Ryan and Salakhutdinov, Ruslan},
        journal={arXiv preprint arXiv:1511.04119},
        year={2015}
    } 

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio. *To appear ICML (2015)*

    @article{Xu2015show,
        title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
        author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1502.03044},
        year={2015}
    }

## License
This repsoitory is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause). It 
is the implementation for our paper [Action Recognition using Visual Attention](http://arxiv.org/abs/1511.04119). The repository uses some code from the project 
[arctic-caption](https://github.com/kelvinxu/arctic-captions) which is originally the implementation for the paper 
[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044) and is also licensed 
under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).
