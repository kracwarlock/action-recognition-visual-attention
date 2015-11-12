import sys
import h5py
import numpy as np
import time

class DataHandler(object):

  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames		# no of timesteps
    self.seq_stride_ = data_pb.stride			# stride for overlap
    self.randomize_ = data_pb.randomize			# randomize their order for training
    self.batch_size_ = data_pb.batch_size		# batch size
    self.fps_ = data_pb.fps
    skip = int(30.0/self.fps_)

    if data_pb.dataset != 'h2mAP':
      labels = self.GetLabels(data_pb.labels_file)	# labels
    else:
      labels = self.GetMAPLabels(data_pb.labels_file)	# multi class labels for mAP

    self.num_frames_ = []
    init_labels_ = []

    num_f = []						# number of frames in each example
    for line in open(data_pb.num_frames_file):
      num_f.append(int(line.strip()))
    assert len(num_f) == len(labels)

    for i in xrange(len(num_f)):
      self.num_frames_.append(num_f[i])
      init_labels_.append(labels[i])

    self.num_videos_ = len(init_labels_)
 
    data = h5py.File(data_pb.data_file,'r')[data_pb.dataset_name]	# load dataset
    self.frame_size_ = data.shape[1]					# 3D cube
    self.dataset_name_ = data_pb.dataset_name

    frame_indices = []
    labels_ = []
    lengths_ = []
    self.dataset_size_ = 0
    start = 0
    self.video_ind_ = {}
    for v, f in enumerate(self.num_frames_):
      end = start + f - self.seq_length_*skip + 1
      if end <= start:					# short length sequences also selected
        end = start+1
      frame_indices.extend(range(start, end, self.seq_stride_))
      for i in xrange(start, end, self.seq_stride_):
        self.video_ind_[i] = v
        labels_.append(init_labels_[v])
        lengths_.append(self.num_frames_[v])
      start += f
    self.dataset_size_ = len(frame_indices)
    print 'Dataset size', self.dataset_size_

    self.frame_indices_ = np.array(frame_indices)	# indices of sequence beginnings
    self.labels_ = np.array(labels_)
    self.lengths_ = np.array(lengths_)
    assert len(self.frame_indices_) == len(self.labels_)
    self.vid_boundary_ = np.array(self.num_frames_).cumsum()
    self.Reset()
    self.batch_data_  = np.zeros((self.seq_length_, self.batch_size_, self.frame_size_), dtype=np.float32)
    if data_pb.dataset != 'h2mAP':
      self.batch_label_ = np.zeros((self.seq_length_, self.batch_size_), dtype=np.int64)
    else:
      self.batch_label_ = np.zeros((self.seq_length_, self.batch_size_, 12), dtype=np.int64)
    self.handler = data

  def GetBatch(self, data_pb, verbose=False):
    skip = int(30.0/self.fps_)
    self.batch_data_  = np.zeros((self.seq_length_, self.batch_size_, self.frame_size_), dtype=np.float32)
    batch_size = self.batch_size_
    n_examples = 0
    for j in xrange(batch_size):
      n_examples += 1
      if verbose:
        sys.stdout.write('\r%d of %d' % (j+1, batch_size))
        sys.stdout.flush()
      start = self.frame_indices_[self.frame_row_]
      label = self.labels_[self.frame_row_]
      length= self.lengths_[self.frame_row_]
      vid_ind = self.video_ind_[start]

      self.frame_row_ += 1
      end = start + self.seq_length_ * skip
      if length >= self.seq_length_*skip:
          self.batch_data_[:,j, :] = self.handler[start:end:skip, :]
      else:
          n = 1 + int((length-1)/skip)
          self.batch_data_[:n,j, :] = self.handler[start:start+length:skip, :]
          self.batch_data_[n:,j, :] = np.tile(self.batch_data_[n-1,j, :],(self.seq_length_-n,1))
      if data_pb.dataset != 'h2mAP':
        self.batch_label_[:,j] = np.tile(label,(1,self.seq_length_))
      else:
        self.batch_label_[:,j,:] = np.tile(label,(self.seq_length_,1))
      if self.frame_row_ == self.dataset_size_:
        #self.Reset()
        break

    if data_pb.dataset=='ucf11':
      self.batch_data_ = self.batch_data_.reshape([self.batch_data_.shape[0],self.batch_data_.shape[1],49,1024],order='F').astype('float32')
    elif data_pb.dataset=='h2mAP':
      self.batch_data_ = self.batch_data_.reshape([self.batch_data_.shape[0],self.batch_data_.shape[1],49,1024],order='F').astype('float32')
    elif data_pb.dataset=='hmdb51gln':
      self.batch_data_ = self.batch_data_.reshape([self.batch_data_.shape[0],self.batch_data_.shape[1],49,1024],order='F').astype('float32')

    self.batch_label_ = self.batch_label_.astype('int64')
    return self.batch_data_, self.batch_label_, n_examples

  def GetSingleExample(self, data_pb, idx, offset=0):
    ### length validation
    num_f = []
    for line in open(data_pb.num_frames_file):
      num_f.append(int(line.strip()))

    #if num_f[idx] < self.seq_length_:
    #    print 'Example is too short'
    #    exit()

    ### data_
    try:
      frames_before = np.cumsum(num_f[:idx],0)[-1]
    except IndexError:
      if idx==0:
        frames_before = 0
      else:
        frames_before = np.cumsum(num_f[:idx],0)[-1]
    start = frames_before + offset                 # inclusive
    end   = frames_before + num_f[idx] - 1         # inclusive
    length= num_f[idx] - offset
    skip = int(30.0/self.fps_)

    data_ = np.zeros((self.seq_length_, 1, self.frame_size_), dtype=np.float32)
    f = h5py.File(data_pb.data_file,'r')

    if length >= self.seq_length_*skip:
      data_[:,0,:] = f[self.dataset_name_][start:start+self.seq_length_*skip:skip, :]
    else:
      n = 1 + int((length-1)/skip)
      self.batch_data_[:n,0, :] = f[self.dataset_name_][start:start+length:skip, :]
      self.batch_data_[n:,0, :] = np.tile(self.batch_data_[n-1,0, :],(self.seq_length_-n,1))

    if data_pb.dataset=='ucf11':
      data_ = data_.reshape([data_.shape[0],data_.shape[1],49,1024],order='F').astype('float32')
    elif data_pb.dataset=='h2mAP':
      data_ = data_.reshape([data_.shape[0],data_.shape[1],49,1024],order='F').astype('float32')
    elif data_pb.dataset=='hmdb51gln':
      data_ = data_.reshape([data_.shape[0],data_.shape[1],49,1024],order='F').astype('float32')

    f.close()

    ### label_
    if data_pb.dataset!='h2mAP':
      labels = self.GetLabels(data_pb.labels_file)
      label  = labels[idx]
      label_ = np.zeros((self.seq_length_, 1), dtype=np.int64)
      label_[:,0] = np.tile(label,(1,self.seq_length_))
    else:
      labels = np.array(self.GetMAPLabels(data_pb.labels_file))
      label  = labels[idx,:]                                     # (12,)
      label_ = np.zeros((self.seq_length_,1,12), dtype=np.int64) # (TS, 1, 12) # 12 classes in hollywood2
      label_[:,0,:] = np.tile(label,(self.seq_length_,1))
    assert len(num_f) == len(labels)

    ### fidx_
    fnames = []
    for line in open(data_pb.vid_name_file):
      fnames.append(line.strip())
    fidx_ = fnames[idx]

    return data_, label_, fidx_

  def GetBatchSize(self):
    return self.batch_size_

  def GetLabels(self, filename):
    labels = []
    if filename != '':
      for line in open(filename,'r'):
        labels.append(int(line.strip()))
    return labels

  def GetMAPLabels(self, filename):
    labels = []
    if filename != '':
      for line in open(filename,'r'):
        labels.append([int(x) for x in line.split(',')])
    return labels

  def GetDatasetSize(self):
    return self.dataset_size_

  def Reset(self):
    self.frame_row_ = 0
    if self.randomize_:
      assert len(self.frame_indices_) == len(self.labels_)
      rng_state = np.random.get_state()
      np.random.shuffle(self.frame_indices_)
      np.random.set_state(rng_state)
      np.random.shuffle(self.labels_)

class TrainProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.num_frames = maxlen
    self.stride = stride
    self.randomize = True
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = '/ais/gobi3/u/shikhar/ucf11/dataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/ucf11/dataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/ucf11/dataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/ucf11/dataset/train_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/train_filename.txt'
      self.dataset_name    = 'features'

class TestTrainProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.num_frames = maxlen
    self.stride = stride
    self.randomize = False
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = '/ais/gobi3/u/shikhar/ucf11/dataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/ucf11/dataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/ucf11/dataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/ucf11/dataset/train_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/train_filename.txt'
      self.dataset_name    = 'features'

class TestValidProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.num_frames = maxlen
    self.stride = stride
    self.randomize = False
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = '/ais/gobi3/u/shikhar/ucf11/dataset/valid_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/ucf11/dataset/valid_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/ucf11/dataset/valid_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/ucf11/dataset/valid_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/test_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/test_filename.txt'
      self.dataset_name    = 'features'

class TestTestProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.num_frames = maxlen
    self.stride = stride
    self.randomize = False
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = '/ais/gobi3/u/shikhar/ucf11/dataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/ucf11/dataset/test_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/ucf11/dataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/ucf11/dataset/test_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/test_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/test_filename.txt'
      self.dataset_name    = 'features'

def main():
  fps = 30
  data_pb = TrainProto(128,30,1,'h2mAP',fps)
  dh = DataHandler(data_pb)
  start      = time.time()
  for i in xrange(dh.dataset_size_/dh.batch_size_):
    x,y,n_ex = dh.GetBatch(data_pb)
    #print x.shape
    #print y.shape
    #print n_ex
    #exit()
  end        = time.time()
  print 'Duration', end-start
  x,y,n_ex = dh.GetBatch(data_pb)
  exit()

if __name__ == '__main__':
  main()

