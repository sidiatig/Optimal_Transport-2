import numpy as np
import ot

import os

import utils
from utils import load_embeddings,write

from sklearn.metrics.pairwise import pairwise_distances,rbf_kernel
#from sinkhorn_pointcloud import sinkhorn_loss



def orthogonal_distances(x,y,pi):
    u,s,vt=np.linalg.svd((x.T).dot(y[pi]))
    return u.dot(vt)



def compute_cost(x,y):
    '''Compute the sim matrix '''
    K_x=rbf_kernel(x,x)
    K_y=rbf_kernel(y,y)
    
    return pairwise_distances(K_x,K_y)




def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    '''For CSLS based module '''
    #xp = get_array_module(m)
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k





main_path='/shares/MRC-CU/Samarajiwa/tanmoy/vecmap/data/embeddings/'
src_file='en.emb.txt'
tgt_file='es.emb.txt'

src_path=os.path.join(main_path,src_file)
tgt_path=os.path.join(main_path,tgt_file)

src_words,x=load_embeddings(src_path)
tgt_words,y=load_embeddings(tgt_path)


#Get the sizes of the matrix
m=x.shape[0]
n=y.shape[0]
k=10

utils.normalize(x,'unit')
utils.normalize(x,'center')
utils.normalize(x,'unit')


utils.normalize(y,'unit')
utils.normalize(y,'center')
utils.normalize(y,'unit')


a,b=np.ones(m)/m,np.ones(n)/n

#Build initial dictionaries
#src_ids=[]
#tgt_ids=[]

sim_size=min(x.shape[0], y.shape[0])

K_X=rbf_kernel(x,x)
K_Y=rbf_kernel(y,y)

sim=pairwise_distances(K_X,K_Y)

lam=4
pis=ot.sinkhorn(a,b,sim,lam)
pi=np.argmax(pis,1)
qs=[]
q_init=orthogonal_distances(x,y,pi)
#qs.append(q_init)
x=x.dot(q_init)
lr=0.025
losses=[]
niters=10

srcfile=open('src_mapped.txt','w')
tgtfile=open('tgt_mapped.txt','w')


for i in range(niters):
    #Compute cost
    sim=compute_cost(x,y)
    #Optimal matching
    pis=ot.sinkhorn(a,b,sim,lam)
    #gradient
    pi=np.argmax(pis,1)
    q=orthogonal_distances(x,y,pi)
    x=x.dot(q)
    wass_loss=np.linalg.norm(x-y)
    losses.append(wass_loss)
    print (wass_loss)
    grad=-2*x.T.dot(y[pi])
    q=q-lr*grad

    u,s,vt=np.linalg.svd((y.T).dot(x))
    w=vt.T.dot(u.T)
    x.dot(w)
    
    if i%5==0:
       lr/=2
       lam/=2
    if i%5==0:
        if losses[i]<=losses[i-1]:

           utils.normalize(x,'unit')
           utils.normalize(x,'center')
           utils.normalize(x,'unit')


           utils.normalize(y,'unit')
           utils.normalize(y,'center')
           utils.normalize(y,'unit')

 
           #write(src_words,x,srcfile)
           #write(tgt_words,y,tgtfile)


          




#srcfile=open('src_mapped.txt','w')
#tgtfile=open('tgt_mapped.txt','w')

write(src_words,x,srcfile)
write(tgt_words,y,tgtfile)


    









