# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:14:57 2024

@author: clayp
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image#, ImageFilter
from scipy.optimize import curve_fit


#%%functions for fitting
def gaussian(r,a,r0,sigma,n):
    return a*np.exp(-((r-r0)/(2*sigma))**2)
#%%looping through images and making arrays of the max values of each column and row
count=106
folder='7-9-24b'
radiic=np.zeros(count)
radiixc=np.zeros(count)
radiiyc=np.zeros(count)

radiig=np.zeros(count)
radiixg=np.zeros(count)
radiiyg=np.zeros(count)

#%% Plotting for a circle
# for k in range(0,count):

#     string=str(k)
#     iimg=Image.open(folder+'/'+string+'.jpg')
#     # blurimg=iimg.filter(ImageFilter.BoxBlur(9))
#     timg=np.array(iimg)[:,:,0] #only red channel


#     #print('avg', avg)
#     xc1,xc2,yc1,yc2= 550,1200, 900,1600
#     timg=timg[xc1:xc2,yc1:yc2]
#     avg=np.average(timg)
#     timg=timg-0.8*avg
    
#     ylim,xlim=np.shape(timg)
#     xmax=np.zeros(xlim)
#     ymax=np.zeros(ylim)
    
#     for i,j in enumerate(xmax): #making an array of the max value of every column
#         #print(i,j,np.max(timg[:,i]))
#         xmax[i]=np.max(timg[:,i])
#         #print(xmax)
    
#     for i,j in enumerate(ymax): #making an array of the max value of every row
#         #print(i,j,np.max(timg[i]))
#         ymax[i]=np.max(timg[i])
#         #print(ymax)
        
#     #thresholding the edges of the beam like it is a circle (it is)
#     mx=0.5*np.mean(np.sort(xmax)[-20:]) #take the avg of the highest values
#     tempx=np.where(xmax>0.5*mx) # figure out where the beam is
#     wx=tempx[0][-1]-tempx[0][0] #width
#     radiixc[k]=0.5*wx 

#     my=0.5*np.mean(np.sort(ymax)[-20:])
#     tempy=np.where(ymax>0.5*my)
#     wy=tempy[0][-1]-tempy[0][0]
#     radiiyc[k]=0.5*wy

#     yfit=np.zeros(ylim)
#     xfit=np.zeros(ylim)
    
#     yfit[tempy[0][0]:tempy[0][-1]]=my
#     xfit[tempx[0][0]:tempx[0][-1]]=mx
    
#     xcoordsfit=np.arange(0,len(xfit))
#     ycoordsfit=np.arange(0,len(yfit))
    
#     xcoords=np.arange(0,len(xmax))
#     ycoords=np.arange(0,len(ymax))
    
#     fwhmx=wx**2
#     fwhmy=wy**2
    
#     #radiik=(np.sqrt(fwhmx+fwhmy))/2
#     #radiic[k]=radiik
#     radiixc[k]=np.sqrt(fwhmx)/2
#     radiiyc[k]=np.sqrt(fwhmy)/2
    
#     #plotting all this for visibility    
#     plt.imshow(timg)

#     plt.plot(xcoords,xmax)
#     plt.plot(xcoordsfit,xfit)

#     plt.plot(ymax,ycoords)
#     plt.plot(yfit,ycoordsfit)
    
#     plt.title(folder+' '+string)
#     plt.show()


#%%gaussian version 
for k in range(0,count):
    string=str(k)
    iimg=Image.open(folder+'/'+string+'.jpg')
    # blurimg=iimg.filter(ImageFilter.BoxBlur(9))
    timg=np.array(iimg)[:,:,0] #only red channel

    avg=np.average(timg)
    timg=timg-0.8*avg
    #print('avg', avg)
    xc1,xc2,yc1,yc2= 550,1200, 900,1600
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
    
    #using all points
    # xfit=xmax
    # yfit=ymax
    # xcoordsfit=xcoords=np.arange(0,xlim)
    # ycoordsfit=ycoords=np.arange(0,ylim)

    #only looking at wings by excluding the center
    #picking out values that arent saturated for x    
    for i,j in enumerate(xmax):
        if j<np.max(xmax):
            continue
        else:
            satst=i
            break
    
    xmaxrev=xmax[::-1]
    for i,j in enumerate(xmaxrev):
        if j<np.max(xmax):
            continue
        else:
            satend=len(xmax)-i
            break
        
    xfit1=xmax[:satst]
    xfit2=xmax[satend:]
    xfit=np.append(xfit1,xfit2)
    
    xcoords=np.arange(0,xlim)    
    xcoordsfit1=xcoords[:satst]
    xcoordsfit2=xcoords[satend:]
    xcoordsfit=np.append(xcoordsfit1,xcoordsfit2)
    
    #now for y
    for i,j in enumerate(ymax):
        if j<np.max(ymax):
            continue
        else:
            satsty=i
            break
    
    ymaxrev=ymax[::-1]
    for i,j in enumerate(ymaxrev):
        if j<np.max(ymax):
            continue
        else:
            satendy=len(ymax)-i
            break
        
    yfit1=ymax[:satsty]
    yfit2=ymax[satendy:]
    yfit=np.append(yfit1,yfit2)
    
    ycoords=np.arange(0,ylim)
    ycoordsfit1=ycoords[:satsty]
    ycoordsfit2=ycoords[satendy:]
    ycoordsfit=np.append(ycoordsfit1,ycoordsfit2)
    
    #fitting/plotting for a gaussian           
    paramx, pcovx = curve_fit(gaussian, xcoordsfit,xfit)
    print('x', paramx)


    paramy, pcovy = curve_fit(gaussian, ycoordsfit,yfit)
    print('y', paramy)
    
    sigmax=paramx[2]
    sigmay=paramy[2]
    
    fwhmx=2*sigmax**2 #should this have a parenthese? (2*sigmax)**2?
    fwhmy=2*sigmay**2
    
    #radiik=(np.sqrt(fwhmx+fwhmy))/2
    #radiig[k]=radiik
    radiixg[k]=np.sqrt(fwhmx)
    radiiyg[k]=np.sqrt(fwhmy)
    
    plt.imshow(timg)

    plt.plot(xcoords,xmax)
    plt.plot(xcoordsfit,gaussian(xcoordsfit,paramx[0],paramx[1],paramx[2],paramx[3]))

    plt.plot(ymax,ycoords)
    plt.plot(gaussian(ycoordsfit,paramy[0],paramy[1],paramy[2],paramy[3]),ycoordsfit)
    
    plt.title(folder+' '+string)
    plt.show()

#%%graphing parameters for radius vs z

photonum=np.arange(0,count) #numbering all of the photos (will tell us how many steps)
#radmirr=10.5/2*10 #cm*10mm/cm
radmirr=125.12/2 #for the new big mirror
#dfrommirr=1.182*10 + 2.54*10 #added one in bc i moved setup
#dfrommirr=7.4*10 #for the big mirror
#dfrommirr=12.75*10 #new big mirror
dfrommirr=100.09 -1.77#new setup for big mirror
drail=8.7*10
dperstep=drail/(count) #how many mm per step

#281 pixels per inch (about) -> 281 pixels/2.54 cm -> 281/25.4 mm -> like 10pixels/mm
pixeltomm=281/(25.4) #pixels/mm

start=radmirr+dfrommirr
#start=6.45*10 +25.4 good for small mirrors
def rayleigh(z,n,w0,z0,lam):
    #lam=000.6328/1.000293 # wavelength in vacuum in mm/n air
    k=np.pi/lam
    zr=k*n*w0**2
    sqrt=1+((z-z0)/zr)**2
    return w0*np.sqrt(sqrt)

def line(x,m,b):
    y=m*x+b
    return y
#%%graphing radii vs z to get focal point:
#radiic=np.sqrt(2*radiixc**2+2*radiiyc**2)/2
radiig=np.sqrt(2*radiixg**2+2*radiiyg**2)/2

#radiiarr=radiixc,radiiyc,radiic,radiixg,radiiyg,radiig
radiiarr=radiixg,radiiyg,radiig
labels=['x gaus, ', 'y gaus, ', 'radii gaus, ']
font1 = {'family':'serif','size':28}

for j, i in enumerate(radiiarr):
    
    photonum300=photonum[0:len(i)]
    photonum1=photonum300[i>0] #we don't want to have any radii equal to 0 since our original array is zeros
    radii1=i[i>0]
    
    thres=20
    r_D=np.abs(np.diff(radii1,axis=0))
    r_bad=np.where(r_D>thres) #we expect points to smoothly transition into each other
    #for i in r_bad[0]:
    #    radii1[i]=np.average(radii1[i-5:i+5])
    
    mask=np.ones(len(radii1), dtype=bool)
    mask[[r_bad]]=False
    
    radiig=radii1#[mask]
    photonumm=photonum1#[mask]
    
    clip=140
    
    radiiclip=radiig[0:clip] #cutting outvalues based on index
    radii2=radiiclip[radiiclip<100]
    radiicorr=radii2/pixeltomm #convert to mm 
    
    photonumclip=photonumm[0:clip]
    photonum2=photonumclip[radiiclip<100]
    x2=start+photonum2*dperstep#[radii<30]*dperstep
    
    fit, cov=curve_fit(rayleigh,x2,radiicorr)
    print(labels[j],'n',fit[0],'wo',fit[1],'zo',fit[2],'lam',fit[3])
    beamwidth2=rayleigh(fit[2],*fit)#min(rayleigh(x2,*fit))
    focpoint2=fit[2]#x2[rayleigh(x2,*fit)==beamwidth2]
    
    #looking for where the slope of the fit is roughly linear by
    #taking the slope of the fit at each point then looking at the acc/second der 
    #and finding where the acc is less than like half the acc
    
    #this is for a longer focal length mirror where you can almost approximate
    #the focal point by using a linear fit 
    
    # sloperfit=np.diff(rayleigh(x2,*fit))
    # imprfit=np.diff(sloperfit)
    # thresh=max(imprfit)
    # maskr=np.where(imprfit>0.9*thresh)

    
    # x2l=x2[0:maskr[0][0]]
    # #x2l=x2[0:10]
    # radiil=radiicorr[0:maskr[0][0]]
    # #radiil=radiicorr[0:10]
    # linefit,linecov=curve_fit(line,x2l,radiil)
    # m,b=linefit
    # x0=-b/m
    
    #idea: iterate the upper limit and see how the x0 changes gkjldsahgkl
    # uplims=np.arange(2,60)
    # x0s=np.zeros(len(uplims))
    # for i,j in enumerate(uplims):
    #     x2ltest=x2[0:j]
    #     radiiltest=radiicorr[0:j]
    #     linefittest,linecovtest=curve_fit(line,x2ltest,radiiltest)
    #     mt,bt=linefittest
    #     x0i=-bt/mt
    #     x0s[i]=x0i
    # print(x0s)
    #ignore. troubleshooting slope and thresholdhing techniques
    # print(imprfit)
    # plt.plot(x2[0:len(sloperfit)],sloperfit)
    # plt.xlabel(labels[j]+'slope', fontdict=font1)
    # plt.show()
    # plt.plot(x2[0:len(imprfit)],imprfit)
    # plt.xlabel(labels[j]+'accel', fontdict=font1)
    # plt.show()

    #plt.plot(x2[0:104][imprfit==thresh],thresh,marker='*',color='blue')
    #plt.plot(x2[0:104][np.where(imprfit>0.5*thresh)],sloperfit[0:105][np.where(imprfit>0.5*thresh)],color='red')
    #plt.plot(x2,rayleigh(x2,*fit))
 
    
    # print(imprfit,thresh)
    # print(np.where(imprfit>thresh))
    # print(np.mean(imprfit))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(8,5),dpi=300)
    plt.scatter(x2,radiicorr, color='coral')
    plt.scatter(x2,-radiicorr, color='coral')
    plt.axhline(y = 0.0, color = 'grey', linestyle = 'dashed')
 
    #plt.plot(focpoint2,beamwidth2, marker='*',color='blue',markersize=18,label='Rayleigh fit: '+str(np.round(focpoint2[0],3))+' mm')
    plt.plot(focpoint2,beamwidth2, marker='*',color='blue',markersize=18,label='Rayleigh fit: '+str(np.round(focpoint2,3))+' mm')
    plt.plot(focpoint2, -beamwidth2,marker='*',color='blue',markersize=18)

    # plt.plot(x0,0,marker='*',color='red',markersize=15,label='Line fit: '+str(x0)+' mm')
    
    plt.plot(x2,rayleigh(x2,*fit),color='blue')
    plt.plot(x2,-rayleigh(x2,*fit),color='blue')

    # plt.plot(x2,line(x2,*linefit),color='blue')
    # plt.plot(x2,-line(x2,*linefit),color='blue')

    # plt.plot(x2[maskr],rayleigh(x2,*fit)[maskr],color='red')
    # plt.plot(x2[maskr],-rayleigh(x2,*fit)[maskr],color='red')
    
    plt.xlabel(labels[j]+'Distance from Mirror (mm)', fontdict=font1)

    plt.ylabel('Width of Beam (mm)', fontdict=font1)
    plt.tick_params(labelsize='20')
    
    plt.legend(fontsize=20,loc='upper left')
    plt.show()

