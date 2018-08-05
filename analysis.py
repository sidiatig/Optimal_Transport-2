import numpy as np
from utils import load_embeddings
import os


main_path='/shares/MRC-CU/Samarajiwa/tanmoy/vecmap/data/embeddings/'
src_file='en.emb.txt'
tgt_file='es.emb.txt'

src_path=os.path.join(main_path,src_file)
tgt_path=os.path.join(main_path,tgt_file)

src_words,x=load_embeddings(src_path)
tgt_words,y=load_embeddings(tgt_path)


analysis_path='/shares/MRC-CU/Samarajiwa/tanmoy/vecmap'
src_files='SRC_MAPPED.EMB'
tgt_files='TRG_MAPPED.EMB'


src_paths=os.path.join(analysis_path,src_files)
tgt_paths=os.path.join(analysis_path,tgt_files)

src_word,xw=load_embeddings(src_paths)
tgt_word,yw=load_embeddings(tgt_paths)

print (xw.shape)
print (x.shape)


print(np.linalg.norm(x-xw))

