### bwf2.py versus bwf.py
Note: bwf2.py is more useful and clean. Look at that one first.

##### bwf2.py breakdown
bwf2.py is a version that contains the techniques that worked the best for me. If you'd like to see more options, look below at the bwf.py breakdown. Otherwise, use bwf2.py.

Update 7/31/2024: Added a function that can convert the digital number to luminosity units. This was obtained from [Pagnutti et al.](https://doi.org/10.1117/1.JEI.26.1.013014), who did a really great analysis of the Raspberry Pi Camera. To use this, uncomment line 120: ``timg=lum(timg)``. To graph the intensity versus z, uncomment lines 242-257.

Let's go through the code. First, we have a technique that fits for a geometric beam (40-110) and a version that fits for a gaussian beam (113-258).
The geometric technique uses a threshold value based on the maximum value pixel in the image to find the edges of an assumed circular beam. This threshold is controlled on lines 69-78.
The gaussian technique can either use all of the points in the array (142-148) or only points that are unsaturated (150-201). One of these needs to be uncommented along with lines 114-140 and 203-240 to get the technique to function.  
Both the geometric and gaussian sections can be uncommented together to allow for comparison between the two. Useful if you're unsure whether your beam is better in the geometric or gaussian regime.

There are some useful parameters and sections of the code to keep in mind: 
* (19) ``count``: This, if used in conjunction with run.py, should be set to the the number of images in the folder (e.g. if there are 108 images in the directory but the highest numbered file is 107.png, set ``count=108``)
* (263) ``radmirr``: This should be the width of the mirror or the width of the base in mm divided by two. For tracking gaussian beams, OAP mirrors have z=0 at the center of the mirror.
* (267) ``dfrommirr``: This should be the distance from your first image to the edge of the mirror as measured in radmirr. Keep the two consistent.
* (268) ``drail``: This is the distance between your first image and your last image aka the length of the rail your setup travels along.
* (272) ``pixeltomm``: this is how many pixels/mm your image produces. Since the goal of my research was not to find the width of the beam itself (which is most likely impossible for a raspberry pi camera), I took a rough measurement by taking a picture of a ruler and converting.
* (302) ``thres``: If your image is blurry or does not have a good gaussian fit, curve_fit will fit a gaussian over the whole image rather than just the beam. When this happens, it will give a very large radius. This line basically says that if there is a change between one point and the next that is greater than this value, cut it out of the array.
* (312) ``clip``: Sometimes values near the focal point can be difficult to fit. This shortens the array.
* (337) ``maskr=np.where(imprfit>0.9*thresh)``: The lines 334-346, 399, and 405-407 allow a user to fit a straight line onto the points. ``maskr`` picks out points that have a rate of change of less than the fraction of the maximum value. This way, a line can be fitted to points that are in a linear range, allowing better estimation of focal length for longer focal length mirrors. Change 0.9 to different values until it gives the desired fit. 

##### bwf.py breakdown
bwf.py (beam waist finder.py) is a version of the code that includes every technique I tried to find the beam edge. The code takes an image, picks out the red channel (since it was used with a red laser) then makes an array of the max values in the x direction and in the y direction separately. It then tries to find the edge of the beam by fitting a gaussian, in the case of a gaussian beam, or by using a threshold, in the case of a geometric beam. Looking at bwf.py, we have a section for thresholding as a geometric beam (line 53-96). We also have various methods for fitting a gaussian (178-206), which include using all of the points (97-102), only using pixels that aren't saturated by removing parts of the array that are 255 (103-112), or by looking only at the wings of the beam and excluding the center by picking out the first saturated point on each side and cutting that section out of the array (113-163). We then have the graphing parameters (208-230) which are how we process the final array of radii. dfrommirror (line 213) should be how far the first image of the mirror is from the edge of the mirror, to coincide with whatever size of the mirror you input into radmirr. The rest should be self explanatory. 

For graphing/fitting the radii, lines 239-248 allow us to cut out points that jump too much. This accounts for errors in fitting the beam radius (e.g. sometimes curve_fit will spit out a radius of 200 mm... not very useful). Next, line 249 allows us to cut off one some of the end of the array. This is sort of useful when the beam curves greatly near the beam waist. 267-279 let us fit a line to our radii, which is useful if we are far away from the focal point as a gaussian beam tends to a geometric beam there. Line 270 determines what points we use to fit the line by setting a threshold for the change in 2nd derivative. This lets us look cut out points where the rayleigh fit changes too greatly. Works well for long focal length mirrors. 
