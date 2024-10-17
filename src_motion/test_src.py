from openscene.run.evaluate import init, find
import open3d as o3d



model, feature_type , val_data_loader=init()
path="AI58_003.ply"
trans=[[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]]
positive='apple'
negative=('bookshelf','table','desk','wall', 'ceiling', 'floor','other')
center, aabb=find(model,feature_type,val_data_loader,path,trans, positive,negative)  