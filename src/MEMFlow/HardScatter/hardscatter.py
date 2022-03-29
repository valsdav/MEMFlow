import ttH
ttH.initialise("./ttH/param_card.dat")


def smatrix_ttH(pdgs, vectors, alphas=0.13):
    proc_id = -1 # if you use the syntax "@X" (with X>0), you can set proc_id to that value (this allows to distinguish process with identical initial/final state.) 
    nhel = -1 # means sum over all helicity
    scale2 = 0.  #only used for loop matrix element. should be set to 0 for tree-level
    
    return ttH.smatrixhel(pdgs, proc_id, vectors, alphas, scale2, nhel)
    
