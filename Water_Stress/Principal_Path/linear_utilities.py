import numpy as np

from scipy.spatial import distance

from matplotlib import pyplot

import time
    

def initMedoids(X, n, init_type, exclude_ids=[]): 
    """
    Initialize NC medoids with init_type rational.

    Args:
        [ndarray float] X: data matrix

        [int] n: number of medoids to be selected
        
        [string] init_type: rational to be used
            'uniform': randomly selected with uniform distribution
            'kpp': k-means++ algorithm

        [ndarray int] exclude_ids: blacklisted ids that shouldn't be selected

    Returns:
        [ndarray int] med_ids: indices of the medoids selected
    """

    N=X.shape[0]
    D=X.shape[1]
    med_ids=-1*np.ones(n,int)

    if(init_type=='uniform'):
        while(n>0):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(med_ids==med_id)==0 and np.count_nonzero(exclude_ids==med_id)==0):
                med_ids[n-1]=med_id
                n = n-1

    elif(init_type=='kpp'):
        accepted = False
        while(not accepted):
            med_id = np.random.randint(0,N)
            if(np.count_nonzero(exclude_ids==med_id)==0):
                accepted = True
        med_ids[0]=med_id

        for i in range(1,n):
            Xmed_dst = distance.cdist(X,np.vstack([X[med_ids[0:i],:],X[exclude_ids,:]]),'sqeuclidean') 
            D2 = Xmed_dst.min(1)
            D2_n = 1.0/np.sum(D2)
            accepted = False
            while(not accepted):
                med_id = np.random.randint(0,N)
                if(np.random.rand()<D2[med_id]*D2_n):
                    accepted = True
            med_ids[i]=med_id
    else:
        raise ValueError('init_type not recognized.')

    return(med_ids)



def getMouseSamples2D(X, n):
    ids = np.empty(n, dtype=int)
    n_sel = [n]

    fig, ax = pyplot.subplots()
    ax.plot(X[:, 1], X[:, 2], '.') ### Water-stress
  #  ax.plot(X[:, 0], X[:, 1], '.')
    # ax.set_xlim([-2000, 4000])
    # ax.set_ylim([-6000, 2000])
   # ax.axis('equal')
    ax.set_title(f'Select {n} points')

    def onclick(ev):
        if n_sel[0] == 0:
            pyplot.close()
            return

        mouseX = np.asarray([ev.xdata, ev.ydata], float)
        if np.any(np.isnan(mouseX)):
            return  # Ignore if clicked outside plot

        id_sel = np.argmin(np.linalg.norm(X[:, :2] - mouseX, axis=1))

        ax.plot(X[id_sel, 0], X[id_sel, 1], 'ro')
        ids[n - n_sel[0]] = id_sel
        n_sel[0] -= 1

        if n_sel[0] == 0:
            ax.set_title('Click to quit or close the figure')
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    pyplot.show()
    fig.canvas.mpl_disconnect(cid)

    return ids



def find_elbow(f):
    """
    Find the elbow in a function f, as the point on f with max distance from the line connecting f[0,:] and f[-1,:]

    Args:
        [ndarray float] f: function (Nx2 array in the form [x,f(x)]) 

    Returns:
        [int]  elb_id: index of the elbow 
    """
    ps = np.asarray([f[0,0],f[0,1]])
    pe = np.asarray([f[-1,0],f[-1,1]])
    p_line_dst = np.ndarray(f.shape[0]-2,float)
    for i in range(1,f.shape[0]-1):
        p = np.asarray([f[i,0],f[i,1]])
        p_line_dst[i-1] = np.linalg.norm(np.cross(pe-ps,ps-p))/np.linalg.norm(pe-ps)
    elb_id = np.argmax(p_line_dst)+1

    return elb_id


def project_point_onto_path(x, W):
    """
    Project a single point x onto the piecewise linear path defined by waypoints W.

    Args:
        x: (d,) ndarray - new data point
        W: (N, d) ndarray - path waypoints

    Returns:
        x_proj: (d,) ndarray - projection of x onto path
        min_dist: float - squared distance from x to x_proj
        segment_id: int - index of the segment on which the projection lies (between W[i] and W[i+1])
    """
    min_dist = np.inf
    x_proj = None
    segment_id = -1

    for i in range(W.shape[0] - 1):
        a = W[i]
        b = W[i + 1]
        ab = b - a
        t = np.dot(x - a, ab) / np.dot(ab, ab)  
        t_clipped = np.clip(t, 0.0, 1.0)
        proj = a + t_clipped * ab
        dist = np.sum((x - proj) ** 2)
        if dist < min_dist:
            min_dist = dist
            x_proj = proj
            segment_id = i

    return x_proj, min_dist, segment_id
