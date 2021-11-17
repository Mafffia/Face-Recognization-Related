#play around with insightface model.
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from lfw import gen_all_pic_labels
from sklearn.cluster import DBSCAN
from insightface.data import get_image as ins_get_image
from sklearn.decomposition import PCA
import hdbscan
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import umap

from mpl_toolkits.mplot3d import Axes3D
# pics,names = gen_all_pic_labels('./lfw_funneled/')
# app = FaceAnalysis()
# app.prepare(ctx_id=0, det_size=(640, 640))
# url1 = pics[0]

# img = cv2.imread(url1)
# faces = app.get(img)
# print((faces[0]['embedding'].shape))
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./url1.jpg", rimg)

#define number of faces loaded <=6000
num = 100
pics,names = gen_all_pic_labels('./lfw_funneled/')
df = pd.DataFrame({'files':pics,'name':names})
df['name'] = pd.Categorical(df.name).codes
df = df.sample(frac=1).reset_index()
#only get first few lines
df = df[0:num]
counts = df.name.value_counts()
print(counts)
print(df.name.value_counts())
print(df)
embs = np.zeros((num,512))
app = FaceAnalysis(name='buffalo_l')
label = []
app.prepare(ctx_id=0, det_size=(640, 640))

# pca = PCA(n_components=50)
# newX = pca.fit_transform(X) 
for i in range(num):
    pic = df.loc[i]['files']
    name = df.loc[i]['name']
    label.append(name)
    print(pic,name)
    img = cv2.imread(pic)
    try:
        face_emb = app.get(img)[0]['embedding']
        embs[i,:] = face_emb[:]
    except:
        embs[i,:] = np.zeros((512))
    

# print("emb shape before trans ",embs.shape)
# embs = pca.fit_transform(embs)
# print("emb after pca",embs.shape)

# face_tsne = TSNE(n_components=2).fit_transform(embs)
reducer = umap.UMAP(random_state=42,n_components=25)
face_umap = reducer.fit_transform(embs)
clustering = list(DBSCAN(min_samples=2,eps=0.2).fit(face_umap).labels_)
# clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=0.05, approx_min_span_tree=True,
#     gen_min_span_tree=False, leaf_size=40,
#     metric='euclidean', min_cluster_size=2, min_samples=None, p=None)
# clusterer = hdbscan.HDBSCAN()
# clustering = list(clusterer.fit(face_umap).labels_)
out_name = []
out_clu = []

for i in range(len(clustering)):
    if(clustering[i]>=0):
        out_name.append(label[i])
        out_clu.append(clustering[i])

df_out = pd.DataFrame({'real':out_name,'cluster':out_clu})
pd.set_option('display.max_columns',1000)
df_out = df_out.sort_values(by='cluster')
# print(df_out.sort_values(by='real'))
for i in df_out.index:
    print(df_out.loc[i]['cluster'] , '  ' , df_out.loc[i]['real'])
print('total clustered points :%d'%len(out_name))
# print(face_umap.shape)

# fig = plt.figure()
# ax = Axes3D(fig)

# font = {'color':'darkred',"size":13,"family":"serif"}
# # plt.style.use("dark_background")
# # plt.figure(figsize=(8,5))
# ax.scatter(face_umap[:,0],face_umap[:,1],face_umap[:,2],c=label,alpha=0.4,cmap='Paired')
# # cbar = plt.colorbar(ticks=range(10))
# # cbar.set_label(label="digit value",fontdict=font)
# plt.show()
# # plt.clim(-0.5,9.5)




