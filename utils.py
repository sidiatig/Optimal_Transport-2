import numpy as np

def load_embeddings(path,max_vocab_size=10000):
    vectors=[]
    words=[]
    
    with open(path) as f:
         for i, line in enumerate(f):
             if i==0:
                dims=line.strip().split()[1]

             else:
                 word,vect=line.rstrip().split(' ', 1)
                 vect = np.fromstring(vect, sep=' ')
                 if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                 #assert word not in word2id
                 #assert vect.shape == (_emb_dim_file,), i
                 words.append(word)
                 vectors.append(vect[None])
             if i>=max_vocab_size:
                break
    vectors=np.array(vectors)
    vectors=vectors[:,0,:]
    return words,vectors




def write(words, matrix, file):
    m=matrix
    #m=np.asmatrix(matrix)
    #m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    #xp = get_array_module(matrix)
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]


def mean_center(matrix):
    #xp = get_array_module(matrix)
    avg = np.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    #xp = get_array_module(matrix)
    norms = np.sqrt(np.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    #xp = get_array_module(matrix)
    avg = np.mean(matrix, axis=1)
    matrix -= avg[:, np.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)












