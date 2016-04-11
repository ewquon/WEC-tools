#!/usr/local/bin/python
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
pi = np.pi

#----------------
# INPUTS
res = 101               # cells per direction
a,b,c = 5.,5.,2.5       # semi-axes of ellipsoid
x0,y0,z0 = 0.,0.,-2.    # location at which to calculate moment of inertia
rho_water = 1025.       # to calculate exact mass
disp_frac = 0.5         # to calculate exact mass

def density_distribution(x,y,z):
    """ x,y,z are cell-centered coordinates of cells on the surface and within the ellipsoid """

    # additional inputs for density distribution
    t = 1.0 *2.54/100   # in-->m, wall thickness
    rho_concr = 2400.   # kg/m^3, for ballast
    rho_steel = 7750.   # kg/m^3, for wall
    rho_base = rho_steel+1 # +1 to identify for debugging
    zbase = 0.0         # m, height from bottom of float to fill with dense material (to shift mass center downward)
    Mtarget = 134172.

    # - uniform density
#    rho = rho_water * disp_frac * np.ones(x.shape)
    rho = np.zeros(x.shape)

    # - set rho for thin-walled vessel
    r2inner = (x/(a-t))**2 + (y/(b-t))**2 + (z/(c-t))**2
    surfcells = r2inner >= 1
    isurf = np.nonzero(surfcells)[0]
    if len(isurf)==0: print 'WARNING: no cells within',t,'in thick wall'
    rho[isurf] = rho_steel #TODO: set wall material

    # - set rho for base material
    #zbase = -c + zbase # shift to float coords
    basecells = (z <= zbase-c) * ~surfcells
    ibase = np.nonzero(basecells)[0]
    if zbase > 0 and len(ibase)==0: print 'WARNING: no cells for base z <=',zbase
    rho[ibase] = rho_base   #TODO: set base material

    # - set rho for ballast
    #   at this point, we can get the weight of all the other components
    #   to calculate how much ballast we actually need to meet our target
    print 'Mtarget =',Mtarget,'kg'
    Mballast = Mtarget - deltaV*np.sum(rho)
    Vtarget = Mballast / rho_concr #TODO: set ballast material
    print '  need',Mballast,'kg of ballast with volume of',Vtarget,'m^3'
    #rhs = Vtarget / (pi*(a-t)*(b-t)*(c-t)) - 2./3. # this does not account for the base material
    zb = (zbase-c)/c
    rhs = Vtarget / (pi*(a-t)*(b-t)*(c-t)) + zb - zb**3/3
    func = lambda zf: zf - zf**3/3 - rhs
    soln = scipy.optimize.newton(func,0) # solve for z/c
    zfill = soln*(c-t)
    print '  fill to z=',zfill
    ballcells = (z <= zfill) * ~basecells * ~surfcells
    ifill = np.nonzero(ballcells)[0]
    rho[ifill] = rho_concr

    # - output diagnostic info
    print 'Density set for', \
            len(isurf),'wall,', \
            len(ibase),'base, and', \
            len(ifill),'ballast cells'
    #******************
    #***** DEBUG ******
    xconcr = x[rho==rho_concr]
    yconcr = y[rho==rho_concr]
    zconcr = z[rho==rho_concr]
    xsteel = x[rho==rho_steel]
    ysteel = y[rho==rho_steel]
    zsteel = z[rho==rho_steel]
    xbase = x[rho==rho_base]
    ybase = y[rho==rho_base]
    zbase = z[rho==rho_base]
    plt.plot(xsteel[ysteel==0],zsteel[ysteel==0],'b+',label='wall')
    plt.plot(xconcr[yconcr==0],zconcr[yconcr==0],'ko',label='ballast')
    plt.plot(xbase[ybase==0],zbase[ybase==0],'rs',label='base')
    plt.legend(loc='best')
    plt.show()
    #******************

    return rho


################################################################################
################################################################################
################################################################################
# execution starts here

print 'ellipsoid dimensions [m]:',a,b,c
print 'Ncells (per dimension) =',res

Vol_exact = 4./3.*pi*a*b*c
M_exact = rho_water*Vol_exact*disp_frac

# setup mesh
y,x,z = np.meshgrid(
        np.linspace(-b,b,res),
        np.linspace(-a,a,res),
        np.linspace(-c,c,res)
        )
R2 = (x/a)**2 + (y/b)**2 + (z/c)**2
inner = np.nonzero(R2 <= 1)
x = x[inner]
y = y[inner]
z = z[inner]

dx = 2*a/(res-1)
dy = 2*b/(res-1)
dz = 2*c/(res-1)
print '  resolution [m]:',dx,dy,dz
deltaV = dx*dy*dz
print '  differential volume [m^3]:',deltaV

# SANITY CHECK #1
Ninside = len(x)
V = deltaV * Ninside
err = np.abs(Vol_exact-V)
print 'exact / calculated volume =',Vol_exact,V,' ERR [%]=',100*err/M_exact

# SANITY CHECK #2
rho = density_distribution(x,y,z)
M = deltaV * np.sum(rho)
err = np.abs(M_exact-M)
print 'exact (UNIFORM DENSITY) / calculated mass =',M_exact,M,' ERR [%]=',100*err/M_exact

#
# calculate moment of inertia
#
x_cm = deltaV * np.sum(rho * x) / M
y_cm = deltaV * np.sum(rho * y) / M
z_cm = deltaV * np.sum(rho * z) / M
print 'calculated center of mass =',x_cm,y_cm,z_cm

#
# calculate moment of inertia
#
r2 = (
        (y-y0)**2 + (z-z0)**2, 
        (x-x0)**2 + (z-z0)**2, 
        (x-x0)**2 + (y-y0)**2
    )
I = np.zeros(3)
for j in range(3):
    I[j] = deltaV * np.sum(r2[j]*rho)
I_exact = M_exact/5.*np.array([b**2+c**2,a**2+c**2,a**2+b**2])
print 'calculated moment of inertia =',I
#print '  diff from analytical [%]:',100*(I-I_exact)/I_exact

