import gpu_lock
import time
import sys

def LockGPU(max_retries=10):
  """ Locks a free GPU board and returns its id. """
  for retry_count in range(max_retries):
    board = gpu_lock.obtain_lock_id()
    if board != -1:
      break
    time.sleep(1)
  if board == -1:
    print 'No GPU board available.'
    sys.exit(1)
  else:
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu'+str(board))
  return board

def FreeGPU(board):
  """ Frees the board. """
  gpu_lock.free_lock(board)

