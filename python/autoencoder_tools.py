import copy
import random
import struct

class dataset(object):
    def __init__(self,file=None):
        self.binary_mode=False
        self.nrecs=0
        self.npoints=0
        self.data=[]
        self.weight=[]
        self.scale=[]
        if file!=None:self.load(file)
    def load(self,filename):
        fp=open(filename,'rb')
        if fp.read(4)=='\x00\x00\x00\x00':
            #binary format file
            self.binary_mode=True
            self.npoints,self.nrecs=struct.unpack('2i',fp.read(8))
            self.data=self.nrecs*[None]
            self.weight=self.nrecs*[None]
            self.scale=self.nrecs*[None]
            for irec in range(0,self.nrecs):
                self.data[irec]=list(struct.unpack('%id'%self.npoints,fp.read(8*self.npoints)))
            self.scale=list(struct.unpack('%id'%self.nrecs,fp.read(8*self.nrecs)))
            self.weight=list(struct.unpack('%id'%self.nrecs,fp.read(8*self.nrecs)))
            fp.close()            
        else:
            fp.close()
            fp=open(filename,'r')
            lines=fp.readlines()
            fp.close()
            words=lines[0].split()
            self.npoints=int(words[0])
            self.nrecs=int(words[1])
            self.data=self.nrecs*[None]
            self.weight=self.nrecs*[None]
            self.scale=self.nrecs*[None]
            for irec in range(0,self.nrecs):
                words=lines[1+irec].split()
                self.data[irec]=self.npoints*[None]
                for ipoint in range(0,self.npoints):
                    self.data[irec][ipoint]=float(words[ipoint])
                self.scale[irec]=float(words[self.npoints])
                self.weight[irec]=float(words[self.npoints+1])
    def write(self,filename):
        if self.binary_mode:
            fp=open(filename,'wb')
            fp.write('\x00\x00\x00\x00')
            fp.write(struct.pack('2i',self.npoints,self.nrecs))
            for irec in range(0,self.nrecs):
                fp.write(struct.pack('%id'%self.npoints,*self.data[irec]))
            fp.write(struct.pack('%id'%self.nrecs,*self.scale))
            fp.write(struct.pack('%id'%self.nrecs,*self.weight))
                     
            fp.close()
        else:
            fp=open(filename,'w')
            fp.write("%06i %04i\n"%(self.npoints,self.nrecs))
            fmt=" ".join(self.npoints*["%8.5f"])
            for irec in range(0,self.nrecs):
                fp.write(fmt%tuple(self.data[irec]))
                fp.write(" %11.5e %11.5e\n"%(self.scale[irec],self.weight[irec]))
            fp.close()
    def gmtform(self,irec,xstart,xstep,scale=False):
        r=[]
        x=xstart
        for ipt in range(0,self.npoints):
            if scale:
                r+=[[x,self.scale[irec]*self.data[irec][ipt]]]
            else:
                r+=[[x,self.data[irec][ipt]]]
            x+=xstep
        return r
    def sortbyweight(self,reverse=False,wtin=None):
        if wtin==None: 
            weights=copy.deepcopy(self.weight)
        else:
            weights=copy.deepcopy(wtin)
        recnums=range(0,self.nrecs)
        if len(weights)!=self.nrecs: raise StandardError,'oops'
        newweight=[]
        newscale=[]
        newdata=[]
        for irec in range(0,self.nrecs):
            if reverse:
                mw=max(weights)
            else:
                mw=min(weights)
            imw=weights.index(mw)
            newweight+=[self.weight[recnums[imw]]]
            newscale+=[self.scale[recnums[imw]]]
            newdata+=[self.data[recnums[imw]]]
            weights.remove(weights[imw])
            recnums.remove(recnums[imw])
        self.weight=newweight
        self.scale=newscale
        self.data=newdata
    def randomise(self):
        newweight=[]
        newscale=[]
        newdata=[]
        left=range(0,self.nrecs)
        while True:
            try:
                ichoice=random.choice(left)
            except IndexError:
                break
            left.remove(ichoice)
            newweight+=[self.weight[ichoice]]
            newscale+=[self.scale[ichoice]]
            newdata+=[self.data[ichoice]]
        self.weight=newweight
        self.scale=newscale
        self.data=newdata
            
    def fit(self,synth,usescale=False):
        fits=[]
        for irec in range(0,self.nrecs):
            dtd=0.
            ete=0.
            for ipt in range(0,self.npoints):
                if usescale:
                    ete+=((self.scale[irec]*self.data[irec][ipt])-(synth.scale[irec]*synth.data[irec][ipt]))**2.
                    dtd+=(self.scale[irec]*self.data[irec][ipt])**2.
                else:
                    ete+=((self.data[irec][ipt])-(synth.data[irec][ipt]))**2.
                    dtd+=(self.data[irec][ipt])**2.
            fits+=[ete/dtd]
        return fits
    def error(self,synth,usescale=False):
        fits=[]
        for irec in range(0,self.nrecs):
            ete=0.
            for ipt in range(0,self.npoints):
                ete+=((self.data[irec][ipt])-(synth.data[irec][ipt]))**2.
            if usescale:
                fits+=[self.scale[irec]*ete]
            else:
                fits+=[ete]
        return fits
    def maxscale(self):
        return max(self.scale)
    def gmtdiff(self,other,irec,xstart,xstep,scale=False):
        r=[]
        x=xstart
        for ipt in range(0,self.npoints):
            if scale:
                r+=[[x,self.scale[irec]*(self.data[irec][ipt]-other.data[irec][ipt])]]
            else:
                r+=[[x,(self.data[irec][ipt]-other.data[irec][ipt])]]
            x+=xstep
        return r
                
