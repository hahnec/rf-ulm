import numpy as np
from scipy.signal import convolve2d

radial_pala = lambda x, yx, w=2: yx + np.array([localize_radial_symmetry(x[p[0]-w:p[0]+w+1, p[1]-w:p[1]+w+1]) if p[0]>w//2 and p[1]>w//2 and p[0]+w//2<x.shape[0] and p[1]+w//2<x.shape[1] else (0,0) for p in yx.astype('int')])

def localize_radial_symmetry(img, fwhmz=3, fwhmx=3):
    
    # Number of grid points
    nz, nx = img.shape

    # Radial symmetry algorithm

    # grid coordinates are -n:n, where Nz (or Nx) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5. Note that z increases "downward"
    zm_onerow = np.arange(-(nz-1)/2.0+0.5, (nz-1)/2.0-0.5+1)
    zm = np.tile(zm_onerow[:,np.newaxis], (1, nx-1))
    xm_onecol = np.arange(-(nx-1)/2.0+0.5, (nx-1)/2.0-0.5+1)
    xm = np.tile(xm_onecol[np.newaxis,:], (nz-1, 1))

    # Calculate derivatives along 45-degree shifted coordinates (u and v) Please refer to Appendix 2 of the publication attached to this code for basis definition
    dIdu = img[:nz-1,1:nx]-img[1:nz,0:nx-1] # Gradient along the u vector
    dIdv = img[:nz-1,:nx-1]-img[1:nz,1:nx] # Gradient along the v vector

    # Smoothing the gradient of the I window
    h = np.ones((3,3))/9
    fdu = convolve2d(dIdu, h, mode='same') # Convolution of the gradient with a simple averaging filter
    fdv = convolve2d(dIdv, h, mode='same')
    dImag2 = fdu**2 + fdv**2 # Squared gradient magnitude

    # Slope of the gradient . Please refer to appendix 2 of the publication attached to this code for basis/orientation
    m = -(fdv + fdu) / (fdu-fdv)

    # Check if m is NaN (which can happen when fdu=fdv). In this case, replace with the un-smoothed gradient.
    NNanm = np.sum(np.isnan(m))
    if NNanm > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu-dIdv)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]

    # If it's still NaN, replace with zero and we'll deal with this later
    NNanm = np.sum(np.isnan(m))
    if NNanm > 0:
        m[np.isnan(m)] = 0

    # Check if m is inf (which can happen when fdu=fdv).
    try:
        m[np.isinf(m)] = 10*np.max(m[~np.isinf(m)])
    except:
        # Replace m with the unsmoothed gradient
        m = (dIdv + dIdu) / (dIdu-dIdv)

    # Calculate the z intercept of the line of slope m that goes through each grid midpoint
    b = zm - m*xm

    # Weight the intensity by square of gradient magnitude and inverse
    # distance to gradient intensity centroid. This will increase the intensity of areas close to the initial guess
    sdI2 = np.sum(dImag2)
    zcentroid = np.sum(dImag2*zm)/sdI2
    xcentroid = np.sum(dImag2*xm)/sdI2
    w = dImag2/np.sqrt((zm-zcentroid)*(zm-zcentroid)+(xm-xcentroid)*(xm-xcentroid))

    return lsradialcenterfit(m, b, w)

# We'll code the least square solution function separately as we could find the solution with another implementation
def lsradialcenterfit(m, b, w):
    # least squares solution to determine the radial symmetry center

    # inputs m, b, w are defined on a grid
    # w are the weights for each point
    wm2p1 = w/(m**2 + 1)
    sw = np.sum(np.sum(wm2p1))
    smmw = np.sum(np.sum(m**2 * wm2p1))
    smw = np.sum(np.sum(m * wm2p1))
    smbw = np.sum(np.sum(m * b * wm2p1))
    sbw = np.sum(np.sum(b * wm2p1))
    det = smw**2 - smmw * sw
    xc = (smbw * sw - smw * sbw) / det    # relative to image center
    zc = (smbw * smw - smmw * sbw) / det  # relative to image center
    return zc, xc
