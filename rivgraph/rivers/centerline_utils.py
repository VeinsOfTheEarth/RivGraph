# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:41:28 2020

@author: Jon
"""
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import split
import scipy.interpolate as si
from scipy import signal
from scipy.interpolate import interp1d

def resample_line(xs, ys, npts=None, nknots=None, k=3):
    """
    Resample a line.

    Resamples a line defined by x,y coordinates such that coordinates are
    evenly-spaced.
    
    If the optional npts parameter is not specified, the
    line will be resampled with the same number of points as the input
    coordinates.
    k refers to order of spline that is fit to coordinates for resampling.

    """
    # FitPack in si.splprep can't handle duplicate points in the line, so remove them
    no_dupes = np.array(np.where(np.abs(np.diff(xs)) +
                        np.abs(np.diff(ys)) > 0)[0], dtype=np.int)
    xs_nd = np.r_[xs[no_dupes], xs[-1]]
    ys_nd = np.r_[ys[no_dupes], ys[-1]]

    # Get indices of knots
    if nknots is None:
        nknots = len(xs_nd)
    knot_indices = np.linspace(0, len(xs_nd), nknots, dtype=np.int)
    knot_indices[knot_indices > len(xs_nd) - 1] = len(xs_nd) - 1
    knot_indices = np.unique(knot_indices)

    # Create spline
    spline, _ = si.splprep([xs_nd[knot_indices], ys_nd[knot_indices]], k=k)

    # Evaluate spline
    if npts is None:
        npts = len(xs)
    t = np.linspace(0, 1, npts)
    resampled_coords = si.splev(t, spline)

    return resampled_coords, spline


def evenly_space_line(xs, ys, npts=None, k=3, s=0):
    """
    Resample a line with evenly spaced coordinates.

    Resamples a curve defined by x,y coordinates such that coordinates are
    evenly-spaced.
    If the optional npts parameter is not specified, the
    line will be resampled with the same number of points as the input
    coordinates.
    k refers to order of spline that is fit to coordinates for resampling.
    xs, ys must be numpy arrays.

    """
    # FitPack in si.splprep can't handle duplicate points in the line, so remove them
    shapely_line = LineString(zip(xs, ys))
    shapely_line = shapely_line.simplify(0)
    xs_nd, ys_nd = np.array(shapely_line.coords.xy[0]), np.array(shapely_line.coords.xy[1])

    # Create spline
    spline, _ = si.splprep([xs_nd, ys_nd], k=k, s=s)

    # Evaluate spline
    if npts is None:
        npts = len(xs)
    t = np.linspace(0, 1, npts)
    resampled_coords = si.splev(t, spline)

    return resampled_coords, spline


def offset_linestring(linestring, distance, side):
    """
    Offset a linestring.

    """

    # Perform the offset
    offset = linestring.parallel_offset(distance, side)

    # Ensure that offset is not a MultiLineString by deleting all but the longest linestring
    if type(offset) is shapely.geometry.multilinestring.MultiLineString:
        ls_lengths = [ls.length for ls in offset]
        offset = offset[ls_lengths.index(max(ls_lengths))]
        print('Multilinestring returned in offset_linestring; clipped to longest but check output.')

    # Ensure offset linestring is oriented the same as the input linestring
    xy_orig_start = (linestring.coords.xy[0][0], linestring.coords.xy[1][0])

    # Get endpoint coordinates of offset linestring
    xy_offset_start = (offset.coords.xy[0][0], offset.coords.xy[1][0])
    xy_offset_end = (offset.coords.xy[0][-1], offset.coords.xy[1][-1])

    if np.sum(np.sqrt((xy_orig_start[0]-xy_offset_start[0])**2 +
                      (xy_orig_start[1]-xy_offset_start[1])**2)) > np.sum(np.sqrt((xy_orig_start[0]-xy_offset_end[0])**2 + (xy_orig_start[1]-xy_offset_end[1])**2)):
        offset = LineString(offset.coords[::-1])

    return offset


def inflection_pts_oversmooth(xs, ys, n_infs):
    """
    Compute inflection points.

    Computes inflection points as the intersection of a line given by
    xs, ys and its (over)smoothed version.

    Parameters
    ----------
    xs : np.array
        x-coordinates of centerline arranged upstream->downstream
    ys : np.array
        y-coordinates of centerline arranged upstream->downstream
    n_infs :
        approx how many inflection points are expected? This sets the
        degree of smoothing. For meandering rivers, n_infs can be
        approximated by the relationship: wavelength = 10W, for average
        width W.

    """
    def generate_smoothing_windows(start, stop, nbreaks, polyorder=3):
        """
        Generate smoothing window sizes.

        Generate an array of window sizes for smoothing. Each value must be
        greater than the polyorder of the smoother and be odd.
        """

        smoothwins = np.linspace(start, stop, nbreaks, dtype=np.int)

        # Window must be greater than polyorder
        smoothwins[smoothwins < polyorder] = polyorder + 1
        # Window must be odd
        smoothwins[smoothwins % 2 == 0] = smoothwins[smoothwins % 2 == 0] + 1
        smoothwins = np.unique(smoothwins)

        return smoothwins

    def smoothing_iterator(xs, ys, smoothing_windows, n_infs, polyorder):

        for i, sw in enumerate(smoothing_windows):

            # Smooth the line's coordinates
            xs_sm = signal.savgol_filter(xs, window_length=sw,
                                         polyorder=polyorder, mode='interp')
            ys_sm = signal.savgol_filter(ys, window_length=sw,
                                         polyorder=polyorder, mode='interp')

            # Conver to shapely objects for intersection detection
            ls = LineString([(x, y) for x, y in zip(xs, ys)])
            ls_sm = LineString([(x, y) for x, y in zip(xs_sm, ys_sm)])

            intersects = ls.intersection(ls_sm)
            if type(intersects) is shapely.geometry.point.Point:  # Points have no length so check
                n_ints = 1
            else:
                n_ints = len(ls.intersection(ls_sm))

            if n_ints < n_infs:
                break

        return smoothing_windows[i-1], smoothing_windows[i]

    # Set polyorder for smoother; could be included as a parameter
    polyorder = 3

    # Find the smoothing window that provides a number of intersections closest
    # to the provided n_infs
    prepost_tolerance = 10  # difference between smoothing window sizes to stop iterating
    prev = 0  # initial minimum smoothing window size
    post = len(xs)/5  # initial maximum smoothing window size
    while post - prev > prepost_tolerance:
        s_windows = generate_smoothing_windows(prev, post, 25, polyorder=3)
        prev, post = smoothing_iterator(xs, ys, s_windows, n_infs, polyorder)

        ## TODO: should add a counter/tracker to ensure n_infs can actually be
        # obtained and avoid an infinite loop

    # Use the optimized smoothing window to smooth the signal
    window = post
    xs_sm = signal.savgol_filter(xs, window_length=window, polyorder=polyorder,
                                 mode='interp')
    ys_sm = signal.savgol_filter(ys, window_length=window, polyorder=polyorder,
                                 mode='interp')

    # Cast coordinates as shapely LineString objects
    ls = LineString([(x, y) for x, y in zip(xs, ys)])
    ls_sm = LineString([(x, y) for x, y in zip(xs_sm, ys_sm)])

    # Compute intersection points between original and oversmoothed centerline
    int_pts = ls.intersection(ls_sm)
    int_coords = np.array([(int_pts[i].coords.xy[0][0],
                            int_pts[i].coords.xy[1][0]) for i in range(len(int_pts))])

    # Map the intersecting coordinates to the indices of the original signal
    idx = []
    for ic in int_coords:
        idx.append(np.argmin(np.sqrt((ic[0]-xs)**2+(ic[1]-ys)**2)))
    idx = np.sort(idx)

#    plt.close('all')
#    plt.plot(xs_sm,ys_sm)
#    plt.plot(xs,ys)
#    plt.plot(xs[idx], ys[idx], '.')
#    plt.axis('equal')

    # Package oversmoothed coordinates for export
    smoothline = np.array([xs_sm, ys_sm])

    return idx, smoothline


def curvars(xs, ys, unwrap=True):

    """
    Compute curvature (and intermediate variables) for a given set of x,y
    coordinates.
    """
    xs = np.array(xs)
    ys = np.array(ys)

    xAi0 = xs[:-1]
    xAi1 = xs[1:]
    yAi0 = ys[:-1]
    yAi1 = ys[1:]

    # Compute angles between x,y nodes
#    A = np.arctan(np.divide(yAi1-yAi0,xAi1-xAi0))
    A = np.arctan2(yAi1-yAi0, xAi1-xAi0)

    if unwrap is not True:
        Acopy = A.copy()

    # Fix phase jumps in angle larger than pi
    A = np.unwrap(A)

    # Compute distance and cumulative distance between nodes
    s, ds = s_ds(xs, ys)
    s = np.delete(s, 0)

    # Compute curvature via central differencing
    # See: http://terpconnect.umd.edu/~toh/spectrum/Differentiation.html
    sd = np.zeros([len(s)])
    sd[0] = s[2]
    sd[1:] = s[0:-1]

    su = np.zeros([len(s)])
    su[0:-1] = s[1:]
    su[-1] = s[-3]

    Ad = np.zeros([len(s)])
    Ad[0] = A[2]
    Ad[1:] = A[0:-1]

    Au = np.zeros([len(s)])
    Au[0:-1] = A[1:]
    Au[-1] = A[-3]

    # Curvatures - checked against Matlab implementation, OK
    C = -np.divide((np.divide(Au-A, su-s)*(s-sd)+np.divide(A-Ad, s-sd)*(su-s)), (su-sd))

    if unwrap is not True:
        Areturn = Acopy
    else:
        Areturn = A

    return C, Areturn, s


def s_ds(xs, ys):

    ds = np.sqrt((np.diff(xs))**2+(np.diff(ys))**2)
    s = np.insert(np.cumsum(ds), 0, 0)

    return s, ds


def smooth_curvatures(C, cvtarget, tolerance=10):
    """
    Smoothes a curvature signal until the coefficient of variation of its
    differenced curvatures is within tolerance percent.
    """
    from scipy.stats import variation

    smoothstep = int(len(C)/100)
    window = smoothstep
    if window % 2 == 0:  # Window must be odd
        window = window + 1

    cv = 10000
    while abs((cv-cvtarget)/cv*100) > tolerance:
        Cs = signal.savgol_filter(C, window_length=window, polyorder=3,
                                  mode='interp')
        cv = variation(np.diff(Cs))

        window = window + smoothstep
        if window % 2 == 0:  # Window must be odd
            window = window + 1

        if window > len(C)/2:
            print('Could not find solution.')
            return Cs

    return Cs


def inflection_points(C):
    """
    Returns the inflection points for an input curvature signal.
    """
    infs1 = np.where(np.logical_and(C[1:] > 0, C[:-1] < 0))[0] + 1
    infs2 = np.where(np.logical_and(C[1:] < 0, C[:-1] > 0))[0] + 1
    infs = np.sort(np.concatenate((infs1, infs2)))

    return infs


def cl_migration_transect_matching(path_matchers, x1, y1, x2, y2, dt, mig_spacing):
    """
    Computes migration between two centerlines with a user-provided geovector
    file that specifies matching transects. A "matching transect" is simply
    a line segment that intersects the same along-stream point on both 
    centerlines. It provides a manual way to determine where points on a
    centerline map to that centerline at a later time. Implemented due to the 
    failure of the ZS method for downstream-translation.
    
    This function works by parameterizing the channel position as a function
    of centerline distance between every pair of matching transects, then 
    evaluating this parameterization at equally-spaced intervals. This procedure
    is done for both centerlines, and the migration vectors are thus the 
    distance between each pair of evaluated points. 
    """
    
    def get_along_cl_distance(cl, pt, origin):
        """
        Returns the upstream segment of a centerline that is split by splitter
        geometry.
        """    
        # Shapely's split function works only at vertices of linestring
        splt = split_precise(cl, pt)    
        
        if len(splt) == 1:
            int_pt = cl.intersection(pt)
            if int_pt.coords.xy[0][0] == origin[0] and int_pt.coords.xy[1][0] == origin[1]:
                dist = 0
            else:
                dist = cl.length
        else:
            for s in splt:
                if s.coords.xy[0][0] == origin[0] and s.coords.xy[1][0] == origin[1]:
                    dist = s.length
                break
        
        return dist
        
    
    def split_precise(linestring, splitter):
        """
        Shapely's split() function will only split LineStrings at vertices.
        This returns the split LineStrings exactly at the point of intersection.
        Assumes a single intersection.
        """
        
        # If there is no intersection, return the linestring
        if linestring.intersects(splitter) is False:
            return linestring
        
        ssplit = split(linestring, splitter)
        split_pt = linestring.intersection(splitter)
        
        splits = []
        for s in ssplit:
            if Point(s.coords[0]).distance(split_pt) < Point(s.coords[-1]).distance(split_pt):
                splits.append(LineString(list(split_pt.coords) + list(s.coords))) 
            else:
                splits.append(LineString(list(s.coords) + list(split_pt.coords))) 
                
        return splits
    
#    import pdb
#    pdb.set_trace()

    # Convert centerlines to LineStrings
    clls1 = LineString([Point(x,y) for x,y in zip(x1, y1)])
    clls2 = LineString([Point(x,y) for x,y in zip(x2, y2)])

    # Read in manual matched indices
    matchers = gpd.read_file(path_matchers)
    origin = (clls1.coords.xy[0][0], clls1.coords.xy[1][0])
    
    # Add matchers at the beginning and end of the centerlines
    m_begin = LineString(((clls1.coords.xy[0][0], clls1.coords.xy[1][0]), (clls2.coords.xy[0][0], clls2.coords.xy[1][0])))
    m_end = LineString(((clls1.coords.xy[0][-1], clls1.coords.xy[1][-1]), (clls2.coords.xy[0][-1], clls2.coords.xy[1][-1])))
    matchers = gpd.GeoDataFrame(geometry=matchers.geometry.values.tolist() + [m_begin] + [m_end], crs=matchers.crs)
    
    # Order matchers from US->DS
    m_s = [get_along_cl_distance(clls1, g, origin) for g in matchers.geometry.values]
    matchers['s'] = m_s
    matchers = matchers.sort_values(by=['s'])

    pts_mig1 = []
    pts_mig2 = []
    all_dists = np.array([])
    # Loop through each pair of transects; note that this assumes they are already
    # in order
    for m_u, m_d, dist_u, dist_d, idx in zip(matchers.geometry.values[0:-1], matchers.geometry.values[1:], matchers.s.values[0:-1], matchers.s.values[1:], matchers.index.values):
        
        # Get centerline segments between matching transects
        # First centerline
        us_split = split_precise(clls1, m_u)
        for uss in us_split:
            if m_d.intersects(uss):
                ds_split = split_precise(uss, m_d)
                ds_pt = m_d.intersection(uss)
                if len(ds_split) == 1 or Point(ds_split[0].coords.xy[0][-1], ds_split[0].coords.xy[1][-1]) == ds_pt:
                    seg1 = ds_split[0]
                else:
                    seg1 = ds_split[1]
                break
                    
        # Second centerline
        us_split = split_precise(clls2, m_u)
        for uss in us_split:
            if m_d.intersects(uss):
                ds_split = split_precise(uss, m_d)
                ds_pt = m_d.intersection(uss)
                if len(ds_split) == 1 or Point(ds_split[0].coords.xy[0][-1], ds_split[0].coords.xy[1][-1]) == ds_pt:
                    seg2= ds_split[0]
                else:
                    seg2 = ds_split[1]
                break
             
        # Interpolate matching points along segments
        n = round(seg1.length/mig_spacing)
        sparam = np.arange(0, n) / n
        pts1 = [seg1.interpolate(sp, normalized=True) for sp in sparam]
        pts2 = [seg2.interpolate(sp, normalized=True) for sp in sparam]
        
        pts_mig1.extend(pts1)
        pts_mig2.extend(pts2)
        
        # Update distances
        all_dists = np.append(all_dists, [dist_u + sparam*(dist_d-dist_u)])
    
    # Add the end points to the list
    pts_mig1.append(Point(clls1.coords.xy[0][-1], clls1.coords.xy[1][-1]))
    pts_mig2.append(Point(clls2.coords.xy[0][-1], clls2.coords.xy[1][-1]))
    all_dists = np.append(all_dists, clls1.length)
            
    # Compute migration rates
    mig_rates = [p1.distance(p2)/dt for p1, p2 in zip(pts_mig1, pts_mig2)]
    
    # Parameterize migration rate as a function of along-centerline distance
    mr_fxn = interp1d(all_dists, mig_rates)
    
    # Evaluate migration rates at the coordinates of the original centerline
    s, _ = s_ds(x1, y1)
    mr_matched = mr_fxn(s)

    return mr_matched, pts_mig1, pts_mig2


