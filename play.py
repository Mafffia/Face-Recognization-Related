import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from lfw import gen_all_pic_labels
from insightface.data import get_image as ins_get_image
from sklearn.decomposition import PCA

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
num = 50
pics,names = gen_all_pic_labels('./lfw_funneled/')
df = pd.DataFrame({'files':pics,'name':names})
df['name'] = pd.Categorical(df.name).codes
print(df.head)
embs = np.zeros((num,512))
app = FaceAnalysis(model='')
label = []
app.prepare(ctx_id=0, det_size=(640, 640))

pca = PCA(n_components=50)
# newX = pca.fit_transform(X) 
for i in range(num):
    pic = df.loc[i]['files']
    name = df.loc[i]['name']
    label.append(name)
    print(pic,name)
    img = cv2.imread(pic)
    face_emb = app.get(img)[0]['embedding']
    embs[i,:] = face_emb[:]

# print("emb shape before trans ",embs.shape)
# embs = pca.fit_transform(embs)
# print("emb after pca",embs.shape)

# face_tsne = TSNE(n_components=2).fit_transform(embs)
reducer = umap.UMAP(random_state=42,n_neighbors=4,n_components=3)
face_umap = reducer.fit_transform(embs)
print(face_umap.shape)

fig = plt.figure()
ax = Axes3D(fig)

font = {'color':'darkred',"size":13,"family":"serif"}
plt.style.use("dark_background")
plt.figure(figsize=(8,5))
ax.scatter(face_umap[:,0],face_umap[:,1],face_umap[:,2],c=label,alpha=0.4,cmap='Paired')
# cbar = plt.colorbar(ticks=range(10))
# cbar.set_label(label="digit value",fontdict=font)
plt.show()
# plt.clim(-0.5,9.5)




