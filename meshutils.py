from camutils import reconstruct
from collections import defaultdict
import numpy as np
import scipy
import pickle
import trimesh

def generateMesh(directory, threshold,camL,camR,boxlimits,trithresh,resultfile):
    
    imprefixL = "{}/frame_C0_".format(directory)
    imprefixR = "{}/frame_C1_".format(directory)
    imprefixCL = "{}/color_C0_".format(directory)
    imprefixCR = "{}/color_C1_".format(directory)
    
    pts2L,pts2R,pts3,color = reconstruct(imprefixL, imprefixCL, imprefixR, imprefixCR,threshold,camL,camR)
    
    #bounding box pruning 
    drop_pts = []
    for i in range(pts3.shape[1]):
        if(not((boxlimits[0] < pts3[0][i] < boxlimits[1]) and (boxlimits[2] < pts3[1][i] < boxlimits[3]) and (boxlimits[4] < pts3[2][i] < boxlimits[5]))):
            drop_pts.append(i)

    pts3 = np.delete(pts3, drop_pts, 1)
    pts2L = np.delete(pts2L, drop_pts, 1)
    pts2R = np.delete(pts2R, drop_pts, 1)
    color = np.delete(color, drop_pts, 1)
    
    #triangle pruning + removing points with no triangles
    trans_pts3 = pts3.T

    #for pts2L first 
    triL = scipy.spatial.Delaunay(pts2L.T)
    drop_simplicesL = []

    #iterate through all the triangles and store the triangles that have edges whose length is longer than trithresh

    for j in range(triL.simplices.shape[0]):
        point1 = trans_pts3[triL.simplices[j][0]]
        point2 = trans_pts3[triL.simplices[j][1]]
        point3 = trans_pts3[triL.simplices[j][2]]
        if((np.linalg.norm(point1 - point2) > trithresh) or (np.linalg.norm(point2 - point3) > trithresh) or (np.linalg.norm(point1 - point3) > trithresh)):
            drop_simplicesL.append(j)

    # now delete those triangles
    triL.simplices = np.delete(triL.simplices, drop_simplicesL, 0)

    #remapping the indicies of the triangles before we delete the 3d points
    tokeep = np.unique(triL.simplices.flatten())
    remapping = np.zeros(pts3.shape[1])
    remapping[tokeep] = np.arange(0,tokeep.shape[0])
    triL.simplices = remapping[triL.simplices]

    # remove any points which are not refenced in any triangle
    pts3 = pts3[:,tokeep]
    color = color[:,tokeep]
    
    
    #now save all the values into a pickle file
    pickle_file = {}
    pickle_file["pts3"] = pts3
    pickle_file["tri"] = triL.simplices
    pickle_file["color"] = color
    fid = open(resultfile, "wb" )
    pickle.dump(pickle_file,fid)
    fid.close()
    
def mesh_smoothing(mesh_pickle, iteration):
    mesh_file = np.load(mesh_pickle,allow_pickle=True)
    pts3 = mesh_file["pts3"]
    tri = mesh_file["tri"]

    trans_pts3 = pts3.T
    for i in range(iteration):
        neighbors = defaultdict(list)
        for triangle in tri:
            for vertice in triangle:
                sub_neighbors = triangle.tolist()
                sub_neighbors.remove(vertice)
                for neighbor in sub_neighbors:
                    neighbors[int(vertice)].append(trans_pts3[int(neighbor)]) 


        for i in range(trans_pts3.shape[0]):
            if(len(neighbors[i]) != 0):
                all_neighbors = np.array(neighbors[i])
                trans_pts3[i] = all_neighbors.mean(axis=0)
    
    pickle_file = {}
    pickle_file["pts3"] = trans_pts3.T
    pickle_file["tri"] = tri
    pickle_file["color"] = mesh_file["color"]
    fid = open(mesh_pickle, "wb" ) 
    pickle.dump(pickle_file,fid)
    fid.close()


def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    f = open(filename,"w");
    f.write('ply\n');
    f.write('format ascii 1.0\n');
    f.write('element vertex %i\n' % X.shape[1]);
    f.write('property float x\n');
    f.write('property float y\n');
    f.write('property float z\n');
    f.write('property uchar red\n');
    f.write('property uchar green\n');
    f.write('property uchar blue\n');
    f.write('element face %d\n' % tri.shape[0]);
    f.write('property list uchar int vertex_indices\n');
    f.write('end_header\n');

    C = (255*color).astype('uint8')
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]));
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close();

