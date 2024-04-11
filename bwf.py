# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:14:57 2024

@author: clayp
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.optimize import curve_fit


#%%functions for fitting
def gaussian(r,a,r0,sigma,n):
    return a*np.exp(-((r-r0)/(2*sigma))**8)
#%%looping through images and making arrays of the max values of each column and row
count=106
folder='4-11-24a'
radii=np.zeros(count)
radiix=np.zeros(count)
radiiy=np.zeros(count)

for k in range(0,count):
    string=str(k)
    iimg=Image.open(folder+'/'+string+'.jpg')
    # blurimg=iimg.filter(ImageFilter.BoxBlur(9))
    timg=np.array(iimg)[:,:,0] #only red channel

    avg=np.average(timg)
    timg=timg-avg
    print('avg', avg)
    xc1,xc2,yc1,yc2= 500,850, 950,1350
    timg=timg[xc1:xc2,yc1:yc2]
    
    
    ylim,xlim=np.shape(timg)
    xmax=np.zeros(xlim)
    ymax=np.zeros(ylim)
    

    
    for i,j in enumerate(xmax): #making an array of the max value of every column
        #print(i,j,np.max(timg[:,i]))
        xmax[i]=np.max(timg[:,i])
        #print(xmax)
    
    for i,j in enumerate(ymax): #making an array of the max value of every row
        #print(i,j,np.max(timg[i]))
        ymax[i]=np.max(timg[i])
        #print(ymax)
        
#%% thresholding again sigh
    # mx=np.mean(np.sort(xmax)[-10:])
    # tempx=np.where(xmax>0.5*mx)
    # wx=tempx[0][-1]-tempx[0][0]
    # radiix[k]=0.5*wx

    # my=np.mean(np.sort(ymax)[-10:])
    # tempy=np.where(ymax>0.5*mx)
    # wy=tempy[0][-1]-tempy[0][0]
    # radiiy[k]=0.5*wy    
#%%fitting to all points
    # xfit=xmax
    # yfit=ymax
    # xcoordsfit=xcoords=np.arange(0,xlim)
    # ycoordsfit=ycoords=np.arange(0,ylim)
    
#%%fitting only to points that aren't saturated
    xfit=xmax[xmax<np.max(xmax)] #accounting for saturation
    yfit=ymax[ymax<np.max(ymax)]

    xcoords=np.arange(0,xlim)
    xcoordsfit=xcoords[xmax<np.max(xmax)]
    
    ycoords=np.arange(0,ylim)
    ycoordsfit=ycoords[ymax<np.max(ymax)]
    
#%%only looking at wings by excluding the center
    #picking out values that arent saturated for x    
    # for i,j in enumerate(xmax):
    #     if j<np.max(xmax):
    #         continue
    #     else:
    #         satst=i
    #         break
    
    # xmaxrev=xmax[::-1]
    # for i,j in enumerate(xmaxrev):
    #     if j<np.max(xmax):
    #         continue
    #     else:
    #         satend=len(xmax)-i
    #         break
        
    # xfit1=xmax[:satst]
    # xfit2=xmax[satend:]
    # xfit=np.append(xfit1,xfit2)
    
    # xcoords=np.arange(0,xlim)    
    # xcoordsfit1=xcoords[:satst]
    # xcoordsfit2=xcoords[satend:]
    # xcoordsfit=np.append(xcoordsfit1,xcoordsfit2)
    
    # #now for y
    # for i,j in enumerate(ymax):
    #     if j<np.max(ymax):
    #         continue
    #     else:
    #         satsty=i
    #         break
    
    # ymaxrev=ymax[::-1]
    # for i,j in enumerate(ymaxrev):
    #     if j<np.max(ymax):
    #         continue
    #     else:
    #         satendy=len(ymax)-i
    #         break
        
    # yfit1=ymax[:satsty]
    # yfit2=ymax[satendy:]
    # yfit=np.append(yfit1,yfit2)
    
    # ycoords=np.arange(0,ylim)
    # ycoordsfit1=ycoords[:satsty]
    # ycoordsfit2=ycoords[satendy:]
    # ycoordsfit=np.append(ycoordsfit1,ycoordsfit2)

#%% diff
   #numpy.diff >> anything bigger than 50. also 
   #consider separating and plotting x and y and seeing
   #get max whole grid then set threshold to half max
    # thres=0.5*np.max(timg)
    # x_D=np.diff(np.max(timg), axis=1)
    # y_D=np.diff(np.max(timg),axis=0)
    
    # x_flat=np.where(x_D>thres)
    # y_flat=np.where(y_D>thres)
    
    # x_wid=x_flat[0][-1]-x_flat[0][0] #this should be # of points in flat part.
    # y_wid=y_flat[0][-1]-y_flat[0][0]

#%%fitting/plotting            
    paramx, pcovx = curve_fit(gaussian, xcoordsfit,xfit)
    print('x', paramx)


    paramy, pcovy = curve_fit(gaussian, ycoordsfit,yfit)
    print('y', paramy)
    
    sigmax=paramx[2]
    sigmay=paramy[2]
    
    fwhmx=2*sigmax**2
    fwhmy=2*sigmay**2
    
    radiik=(np.sqrt(fwhmx+fwhmy))/2
    radii[k]=radiik
    radiix[k]=fwhmx
    radiiy[k]=fwhmy
    
    plt.imshow(timg)

    plt.plot(xcoords,xmax)
    plt.plot(xcoordsfit,gaussian(xcoordsfit,paramx[0],paramx[1],paramx[2],paramx[3]))

    plt.plot(ymax,ycoords)
    plt.plot(gaussian(ycoordsfit,paramy[0],paramy[1],paramy[2],paramy[3]),ycoordsfit)
    
    plt.title(folder+' '+string)
    plt.show()

#%%graphing parameters

photonum=np.arange(0,count)
radmirr=10.5/2*10 #cm*10mm/cm
dfrommirr=1.182*10 #+ 2.54*10 #added one in bc i moved setup
drail=8.7*10

#281 pixels per inch (about) -> 281 pixels/2.54 cm -> 281/25.4 mm -> like 10pixels/mm
pixeltomm=281/(25.4) #pixels/mm

dperstep=drail/(count) #how many mm per step
#start=radmirr+dfrommirr
start=6.45*10
def rayleigh(z,w0,n,z0,lam):
    #lam=000.6328/1.000293 # wavelength in vacuum in mm/n air
    k=np.pi/lam
    zr=k*n*w0**2
    sqrt=1+((z-z0)/zr)**2
    return w0*np.sqrt(sqrt)

#%%graphing combined radii:
radiiarr=radiix,radiiy,radii
for i in radiiarr:
    photonum1=photonum[i>0] #we don't want to have any radii equal to 0 since our original array is zeros
    radii1=i[i>0]
    print(radiiy)
    thres=5
    r_D=np.diff(radii1,axis=0)
    r_bad=np.where(r_D>thres) #we expect points to smoothly transition into each other
    
    mask=np.ones(len(radii1), dtype=bool)
    mask[[r_bad]]=False
    
    radiig=radii1[mask]
    photonumm=photonum1[mask]
    
    radiiclip=radiig[0:90] #cutting outvalues based on index
    radii2=radiiclip[radiiclip<50]
    radiicorr=radii2/pixeltomm #convert to mm 
    photonumclip=photonumm[:90]
    photonum2=photonumclip[radiiclip<50]
    x2=start+photonum2*dperstep#[radii<30]*dperstep
    
    fit, cov=curve_fit(rayleigh,x2,radiicorr)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(8,5),dpi=300)
    plt.scatter(x2,radiicorr, color='coral')
    plt.scatter(x2,-radiicorr, color='coral')
    plt.axhline(y = 0.0, color = 'grey', linestyle = 'dashed')
    
    beamwidth2=min(rayleigh(x2,*fit))
    focpoint2=x2[rayleigh(x2,*fit)==beamwidth2]
    #plt.axvline(focpoint2, color='red', linestyle='-.')
    
    plt.plot(focpoint2,beamwidth2, marker='*',color='blue',markersize=18,label='Focal Point at '+str(np.round(focpoint2[0]/10,3))+' cm')
    plt.plot(focpoint2, -beamwidth2,marker='*',color='blue',markersize=18)
    
    plt.plot(x2,rayleigh(x2,*fit),color='blue')
    plt.plot(x2,-rayleigh(x2,*fit),color='blue')
    
    font1 = {'family':'serif','size':28}
    plt.xlabel(str(i)+'Distance from Mirror (mm)', fontdict=font1)
    plt.ylabel('Width of Beam (mm)', fontdict=font1)
    plt.tick_params(labelsize='20')
    
    plt.legend(fontsize=20,loc='upper left')


