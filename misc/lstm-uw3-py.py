# coding: utf-8
from pylab import imshow,show,axis,zeros,rand
import matplotlib
matplotlib.rcParams["image.cmap"] = "afmhot"
matplotlib.rcParams["image.interpolation"] = "none"
import clstm
import h5py
import numpy as np



index = 5
h5 = h5py.File("uw3-dew.h5","r")
imshow(h5["images"][index].reshape(*h5["images_dims"][index]).T)
print h5["transcripts"][index]

# # LSTM + CTC Training 

# 
# All input vectors need to have the same length, so we just take that off the first vector in the dataset. The number of outputs can be taken from the codec.

ninput = int(h5["images_dims"][0][1])
noutput = len(h5["codec"])
print ninput,noutput


# Let's create a small bidirectional LSTM network.
net = clstm.make_net_init("bidi","ninput=%d:nhidden=50:noutput=%d"%(ninput,noutput))
net.setLearningRate(1e-4,0.9)
print clstm.network_info(net)

index = 22
xs = np.array(h5["images"][index].reshape(-1,48,1),'f')
transcript = h5["transcripts"][index]
imshow(xs.reshape(-1,48).T,cmap=matplotlib.cm.gray)

# Note that all sequences (including `xs`) in clstm are of rank 3, with indexes giving the time step, the feature dimension, and the batch index, in order.
# The output from the network is a vector of posterior probabilities at each time step.

net.inputs.aset(xs)
net.forward()
pred = net.outputs.array()
imshow(pred.reshape(-1,noutput).T, interpolation='none')


# Target arrays are similar to the output array but may have a different number of timesteps. They are aligned with the output using CTC.


def mktarget(transcript,noutput):
    N = len(transcript)
    target = zeros((2*N+1,noutput),'f')
    assert 0 not in transcript
    target[0,0] = 1
    for i,c in enumerate(transcript):
        target[2*i+1,c] = 1
        target[2*i+2,0] = 1
    return target


target = mktarget(transcript,noutput)
imshow(target.T)


# The CTC alignment now combines the network output with the ground truth.

seq = clstm.Sequence()
seq.aset(target.reshape(-1,noutput,1))
aligned = clstm.Sequence()
clstm.seq_ctc_align(aligned,net.outputs,seq)
aligned = aligned.array()
imshow(aligned.reshape(-1,noutput).T, interpolation='none')


# Next, we take the aligned output, subtract the actual output, set that as the output deltas, and the propagate the error backwards and update.

# In[14]:

deltas = aligned - net.outputs.array()
net.d_outputs.aset(deltas)
net.backward()
net.update()


# If we repeat these steps over and over again, we eventually end up with a trained network.

# In[15]:

for i in range(10):
    index = int(rand()*len(h5["images"]))
    print i,index
    xs = np.array(h5["images"][index].reshape(-1,ninput,1),'f')
    transcript = h5["transcripts"][index]
    net.inputs.aset(xs)
    net.forward()
    pred = net.outputs.array()
    target = mktarget(transcript,noutput)
    seq = clstm.Sequence()
    seq.aset(target.reshape(-1,noutput,1))
    aligned = clstm.Sequence()
    clstm.seq_ctc_align(aligned,net.outputs,seq)
    aligned = aligned.array()
    deltas = aligned - net.outputs.array()
    net.d_outputs.aset(deltas)
    net.backward()
    net.update()


# In[16]:


imshow(xs.reshape(-1,ninput).T)
show()

# In[17]:

def log10max(a,eps=1e-3):
    return np.log10(np.maximum(a,eps))


# In[18]:


imshow(xs.reshape(-1,ninput)[:200].T)
imshow(pred.reshape(-1,noutput)[:200].T)


# Let's write a simple decoder.

# In[19]:

classes = np.argmax(pred,axis=1)[:,0]
print classes[:100]


# When we turn this back into a string using a really simple decoder, it doesn't come out too well, but we haven't trained that long anyway. In addition, this decoder is actually very simple

# In[20]:

codes = classes[(classes!=0) & (np.roll(classes,1)==0)]
chars = [chr(h5["codec"][c]) for c in codes]
print "".join(chars)


# Let's wrap this up as a function:

# In[21]:

def decode1(pred):
    classes = np.argmax(pred,axis=1)[:,0]
    codes = classes[(classes!=0) & (np.roll(classes,1)==0)]
    chars = [chr(h5["codec"][c]) for c in codes]
    return "".join(chars)
decode1(pred)


# Here is another idea for decoding: look for minima in the posterior of the epsilon class and then return characters at those locations:

# In[22]:

from scipy.ndimage import filters
def decode2(pred,threshold=.5):
    eps = filters.gaussian_filter(pred[:,0,0],2,mode='nearest')
    loc = (np.roll(eps,-1)>eps) & (np.roll(eps,1)>eps) & (eps<threshold)
    classes = np.argmax(pred,axis=1)[:,0]
    codes = classes[loc]
    chars = [chr(h5["codec"][c]) for c in codes]
    return "".join(chars)    
decode2(pred)


# It's often useful to look at this in the log domain. We see that the classifier still has considerable uncertainty.

# In[23]:

imshow(log10max(pred.reshape(-1,noutput)[:200].T))


# The aligned output looks much cleaner.

# In[24]:

imshow(aligned.reshape(-1,noutput)[:200].T)


# In[25]:

imshow(log10max(aligned.reshape(-1,noutput)[:200].T))
# We can also decode the aligned outut directly.
# In[26]:

print decode1(aligned)
print decode2(aligned,0.9)
# There is a better decoder in the CLSTM library.
