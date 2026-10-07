import compare_run as cr
def clear(T):
    d,r,ox,oy=cr._signed_wall_dist(6)
    nl,nw=int(2*cr.FP_HALF_LENGTH/cr.FP_SPACING)+1,int(2*cr.FP_HALF_WIDTH/cr.FP_SPACING)+1
    lx,wy=np.linspace(-cr.FP_HALF_LENGTH,cr.FP_HALF_LENGTH,nl),np.linspace(-cr.FP_HALF_WIDTH,cr.FP_HALF_WIDTH,nw)
    b=np.vstack([np.c_[lx,np.full(nl,cr.FP_HALF_WIDTH)],np.c_[lx,np.full(nl,-cr.FP_HALF_WIDTH)],np.c_[np.full(nw,cr.FP_HALF_LENGTH),wy],np.c_[np.full(nw,-cr.FP_HALF_LENGTH),wy]])
    c,s=np.cos(T[:,2])[:,None],np.sin(T[:,2])[:,None]
    qx=T[:,0:1]+b[None,:,0]*c-b[None,:,1]*s;qy=T[:,1:2]+b[None,:,0]*s+b[None,:,1]*c
    col=np.clip(((qx-ox)/r).astype(int),0,d.shape[1]-1);row=np.clip(((qy-oy)/r).astype(int),0,d.shape[0]-1)
    return d[row,col].min(1)
