import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
import opencsp.app.sofast.lib.sofast_debug_figure_support as sdfs
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.LineXY import LineXY
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


def calc_mask_raw(
    mask_images: np.ndarray,
    hist_thresh: float = 0.5,
    filt_width: int = 9,
    filt_thresh: int = 4,
    thresh_active_pixels: float = 0.05,
) -> np.ndarray:
    """
    Calculates the active pixels given a light and a dark image.

    Parameters
    ----------
    mask_images : np.ndarray
        MxNx2 array. Image 0 is dark image, image 1 is light image.
    hist_thresh : float, (-inf, inf)
        Fraction between min and max histogram values to define difference
        image threshold.
    filt_width : int, [1, inf)
        Side width of square filter area in pixels.
    filt_thresh : int, [0, inf)
        Minimum number of pixels in filter area to keep.
    thresh_active_pixels : float, [0, 1]
        Minimum fraction of total pixels needed in mask.

    Returns
    -------
    np.ndarray
        2D active pixel mask, bool.

    """
    # Define constants
    N_BINS_IMAGE = 100  # Number of bins to create histogram of image pixels
    N_PEAK_STEP = 10  # Width of steps to take when finding dark and light peaks in image histogram
    HIST_PEAK_THRESH = 0.002  # Min height of difference image histogram to consider a peak.

    # Create delta image
    delta = mask_images[..., 1].astype(np.float32) - mask_images[..., 0].astype(np.float32)

    # Check if only two values exist (light and dark regions)
    if np.unique(delta).size == 2:
        mask_thresh = delta > 0
    else:
        # Calculate histogram of delta image
        hist, edges = np.histogram(delta.flatten(), bins=N_BINS_IMAGE, density=True)

        # Make sure first and last values of histogram are zero
        hist = np.concatenate([[0], hist, [0]])
        bin_step = edges[1] - edges[0]
        edges = np.concatenate([[edges[0] - bin_step], edges, [edges[-1] + bin_step]])

        # Find two peaks in histogram (light and dark regions)
        for dist in np.arange(N_PEAK_STEP, N_BINS_IMAGE, N_PEAK_STEP):
            peaks = find_peaks(x=hist, height=HIST_PEAK_THRESH, distance=dist)[0]
            if len(peaks) == 2:
                break
        if len(peaks) != 2:
            raise ValueError("Not enough distinction between dark and light pixels in mask images.")

        # Calculate minimum between two peaks
        idx_hist_min = np.argmin(hist[peaks[0] : peaks[1]]) + peaks[0]

        # Find index of histogram that is "hist_thresh" the way between the min and max
        thresh_hist_min = edges[idx_hist_min + 1]
        thresh_hist_max = edges[peaks[1]]
        thresh = thresh_hist_min + (thresh_hist_max - thresh_hist_min) * hist_thresh

        # Calculate threshold mask
        mask_thresh = delta > thresh

    # Filter to remove small active areas outside of main mask area
    k = np.ones((filt_width, filt_width), dtype=np.float32) / float(filt_width**2)
    mask_filt = cv.filter2D(mask_thresh.astype(np.float32), -1, k) > float(filt_thresh / (filt_width**2))

    # Combine both masks
    mask_raw = np.logical_and(mask_filt, mask_thresh)

    # Check for enough active pixels
    thresh_active_pixels = int(mask_raw.size * thresh_active_pixels)
    if mask_raw.sum() < thresh_active_pixels:
        lt.error_and_raise(ValueError, f"Mask contains less than {thresh_active_pixels:d} active pixels.")

    # Return raw, unprocessed mask
    return mask_raw


def keep_largest_mask_area(mask: np.ndarray) -> np.ndarray:
    """
    Keeps the largest continuous mask area.

    Parameters
    ----------
    mask : np.ndarray
        Input 2D mask image.

    Returns
    -------
    np.ndarray
        Output mask, same shape as input mask.

    """
    # Find contours of each cluster in mask
    cnts = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    # Find largest contour
    cnt = max(cnts, key=cv.contourArea)

    # Draw largest contour and fill
    mask_processed = np.zeros(mask.shape, dtype=np.uint8)
    cv.drawContours(mask_processed, [cnt], -1, 255, cv.FILLED)

    # Format as boolean
    mask_processed = mask_processed.astype(bool)

    return mask_processed


def centroid_mask(mask: np.ndarray) -> Vxy:
    """
    Calculates the centroid of a mask

    Parameters
    ----------
    mask : np.ndarray
        2D boolean mask.

    Returns
    -------
    Vxy
        Location of XY mask centroid.

    """
    y, x = np.where(mask)
    xs = x.mean()
    ys = y.mean()

    return Vxy((xs, ys))


def edges_from_mask(mask: np.ndarray) -> Vxy:
    """
    Finds all mask edges via convolution.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean mask.

    Returns
    -------
    Vxy
        Location of all XY mask edges.

    """
    # Define kernels
    ks = [
        np.array([[-1], [1]], np.float32),
        np.array([[1], [-1]], np.float32),
        np.array([[-1, 1]], np.float32),
        np.array([[1, -1]], np.float32),
    ]

    # Find edges
    mask_edges = [(cv.filter2D(mask.astype(np.float32), -1, k) == 1)[..., np.newaxis] for k in ks]
    mask_edges = np.concatenate(mask_edges, 2)
    mask_edge = mask_edges.sum(2).astype(bool)

    # Convert to indices
    ys, xs = np.where(mask_edge)
    xs = xs.astype(np.float32)[np.newaxis, :] - 0.5
    ys = ys.astype(np.float32)[np.newaxis, :] - 0.5

    Puv_edges = Vxy((xs, ys))

    return Puv_edges


def refine_mask_perimeter(
    debug: DebugOpticsGeometry,
    debug_mask_img: np.ndarray | None,
    loop_outline_exp: LoopXY,
    Puv_edges: Vxy,
    d_ax: float,
    d_perp: float,
) -> LoopXY:
    """
    Given mask edge points and an expected 2D PERIMETER region, this function refines the perimeter region.

        1. Each line of the region is converted to a rectangular area of
           interest.
        2. All points within the area of interest are fit to a line.
        3. A new region is formed from these fit lines.

    NOTE: This is optimized to find mask perimeters, not facet perimeters.
    Use refine_facet_corners for facets.

    Parameters
    ----------
    debug_cls : DebugOpticsGeometry
        Carries parameters controlling debugging output.
    debug_mask_img: np.ndarray
        Image of mask to provide visual context for debugging figures.
    loop_outline_exp : LoopXY
        Expected perimeter 2D region.
    Puv_edges : Vxy
        Mask 2D edge points.
    d_ax/d_perp
        See rectangle_loop_from_two_points for more information.

    Returns
    -------
    LoopXY
        Refined perimeter.

    """
    neighbor_dist = 2  # pixels
    lines = []
    for idx in range(len(loop_outline_exp)):
        # Get two points
        p1 = loop_outline_exp.vertices[idx]
        p2 = loop_outline_exp.vertices[np.mod(idx + 1, len(loop_outline_exp))]

        # Get bounding rectangle region
        rectangle_loop = rectangle_loop_from_two_points(p1, p2, d_ax, d_perp)

        # Find points in loop
        pts_mask = rectangle_loop.is_inside(Puv_edges)
        line = LineXY.fit_from_points(Puv_edges[pts_mask], neighbor_dist=neighbor_dist)
        lines.append(line)

        # Debugging output for diagnosis
        if debug.debug_active:
            if debug_mask_img is None:
                lt.warn('In refine_mask_perimeter(), skipping debugging figures because a mask image was not provided.')
            else:
                # Setup figure.
                figure_title = f"Loop Edge {idx} Analysis"
                fig_rec = sdfs.start_debug_image_figure(figure_title)
                # Mask
                fig_rec.view.imshow(debug_mask_img, cmap="gray")
                # All edge points.
                fig_rec.view.axis.scatter(*Puv_edges.data, marker=".", c='pink', s=0.8)
                # Input loop to refine.
                fig_rec.view.draw_pq_list(
                    loop_outline_exp.as_xy_list(), close=True, style=rcps.default(marker='arrow', color='green')
                )
                # Bounding rectangle.
                fig_rec.view.draw_pq_list(rectangle_loop.as_xy_list(), close=True, style=rcps.outline(color='orange'))
                # Edge points within rectangle.
                relevant_pts = Puv_edges[pts_mask]
                fig_rec.view.axis.scatter(*relevant_pts.data, marker=".", c='red', s=0.8)
                # Fit line.
                # if line.A > line.B:
                #     # A line with B=0 implies a vertical constant-x line.
                #     # So we intersect with the horizontal box boundaries.
                # line_pt_1 = (750, 1000)
                # line_pt_2 = (1250, 800)
                # fig_rec.view.draw_pq_list([line_pt_1, line_pt_2], style=rcps.outline(color='green'))
                x_min = 0
                x_max = debug_mask_img.shape[1] - 1
                y_min = 0
                y_max = debug_mask_img.shape[0] - 1
                line.draw_in_axis_aligned_box(
                    (x_min, x_max, y_min, y_max), ax=fig_rec.view.axis, style=rcps.outline(color='magenta')
                )
                # Save and close figure.
                sdfs.finish_debug_image_figure(figure_title, 'image', fig_rec, debug)

    # Create updated region
    return LoopXY.from_lines(lines)


def keep_closest_points(
    p1: Vxy, p2: Vxy, Puv_edge: Vxy, Puv_cent: Vxy, step: float, d_perp: float, frac_keep: float
) -> Vxy:
    """
    Keeps points closest to centroid along direction perpendicular to line
    drawn between p1 and p2. This is used to find the edge of a facet where
    multiple parallel edges may exist close to each other.

    Parameters
    ----------
    p1 : Vxy
        Starting point.
    p2 : Vxy
        Ending point.
    Puv_edge : Vxy
        Mask edge points.
    Puv_cent : Vxy
        Centroid location of facet.
    step : float
        Length of search regions, pixels.
    d_perp: float
        Perpendicular padding distance on search region.
    frac_keep : float
        Fraction of pixels in search region to keep.

    Returns
    -------
    Vxy
        Filtered subset of original points.

    """
    # Step along line axis, create sub-region
    dist = (p1 - p2).magnitude()[0]  # pixels
    steps = np.arange(0, dist, step)  # pixels
    steps[-1] = dist

    Vuv_ax = (p2 - p1).normalize()
    R = np.array([[0, -1], [1, 0]])
    Vuv_perp = Vuv_ax.rotate(R)

    Puv_line = []
    for idx in range(steps.size - 1):
        # Get step distances
        d1 = steps[idx]
        d2 = steps[idx + 1]

        # Get step start/end points
        sub_p1 = p1 + Vuv_ax * d1
        sub_p2 = p1 + Vuv_ax * d2

        # Create step region
        sub_loop = rectangle_loop_from_two_points(sub_p1, sub_p2, 0, d_perp)

        # Mask points in that region
        pts_mask = sub_loop.is_inside(Puv_edge)
        pts_sub = Puv_edge[pts_mask]
        n_keep = int(pts_mask.sum() * frac_keep)

        # Keep N points closest to centroid in region in perpendicular direction
        dists = np.abs((pts_sub - Puv_cent).dot(Vuv_perp))
        idxs = np.argsort(dists)[:n_keep]
        Puv_line.append(pts_sub.data[:, idxs])

    return Vxy(np.concatenate(Puv_line, axis=1))


def refine_facet_corners(
    Puv_facet_corns_exp: Vxy, Puv_cent: Vxy, Puv_edges: Vxy, step: float, d_perp: float, frac_keep: float
) -> LoopXY:
    """
    Refines the locations of the facet corners using only points closest to the
    facet centroid. Returns a refined 2D region of the facet.

    Parameters
    ----------
    Puv_facet_corns_exp : Vxy
        Expected locations of facet corners in mask image.
    Puv_cent : Vxy
        Centroid location of facet in mask image.
    Puv_edges : Vxy
        Mask edge points.
    step/d_perp/frac_keep
        See keep_closest_points for more details

    Returns
    -------
    LoopXY
        Refined location of facet outline.

    """
    # Loop over each edge
    num_corns = len(Puv_facet_corns_exp)
    lines = []
    for idx in range(num_corns):
        # Get start and end point
        p1 = Puv_facet_corns_exp[idx]
        p2 = Puv_facet_corns_exp[np.mod(idx + 1, num_corns)]

        # Keep points closest to centroid, stepping axially between points
        Puv_active = keep_closest_points(p1, p2, Puv_edges, Puv_cent, step, d_perp, frac_keep)

        # Fit active points to line
        lines.append(LineXY.fit_from_points(Puv_active))

    # Create region from lines
    return LoopXY.from_lines(lines)


def unwrap_phase(signal: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Unwraps phase from signal data with four phase shifts.

    Parameters
    ----------
    signal : nxN ndarray
        Signal data where n = (4 * ps.size) and N = number of sample points.
    ps : 1d array
        Fringe periods (widths of screen)

    Returns
    -------
    ndarray
        1d array. Pixel position on screen, screen widths

    """
    for idx, p in enumerate(ps):
        c1 = signal[4 * idx, :].astype(np.float32)
        c2 = signal[4 * idx + 1, :].astype(np.float32)
        c3 = signal[4 * idx + 2, :].astype(np.float32)
        c4 = signal[4 * idx + 3, :].astype(np.float32)

        # Calculate current phase
        phase = np.arctan2(c2 - c4, c1 - c3)  # radians
        w = np.mod(phase, 2 * np.pi) / (p * 2 * np.pi)  # screen widths

        if idx == 0:
            # First iteration
            x = np.copy(w)  # screen widths
        else:
            f = 1 / p
            A = x - f / 2  # screen widths
            wa = np.mod(A + f, f)  # screen widths
            x = A + np.mod(w - wa + f, f)  # screen widths
    return x


def calculate_active_pixels_vectors(mask: np.ndarray, camera: Camera) -> Uxyz:
    """
    Calculates active pixel pointing directions in camera coordinates.

    Parameters
    ----------
    mask : np.ndarray
        Mask of active pixels.
    camera : Camera
        Camera object.

    Returns
    -------
    Uxyz
        3D unit vectors, pixel pointing direcitons in camera coordinates.

    """
    # Get active pixels in mask
    pixels_y, pixels_x = np.where(mask)
    pixels = Vxy((pixels_x, pixels_y))

    # Calculate pixel pointing directions
    u_active_pixel_pointing_cam = camera.vector_from_pixel(pixels)  # camera coordinates

    return u_active_pixel_pointing_cam  # camera coordinates


def rectangle_loop_from_two_points(p1: Vxy, p2: Vxy, d_ax: float, d_perp: float) -> LoopXY:
    """
    Creates a rectangular loop from two points, and two distances.

    Parameters
    ----------
    p1/p2 : Vxy
        Two points to define rectangle directions.
    d_ax : float
        Distance before/after p1/p2 (along p1-p2 axis) to add to rectangle
        length.
    d_perp : float
        Distance perpendicular to p1-p2 axis to add to rectangle width.

    Returns
    -------
    LoopXY
        Rectangular loop.

    """
    # Calculate axial and perpendicular directions
    v_axial = (p2 - p1).normalize()

    rot_mat_90 = np.array([[0, 1], [-1, 0]])
    v_perp = v_axial.rotate(rot_mat_90)

    # Create points
    points = []
    points.append(p1 - (v_axial * d_ax) - (v_perp * d_perp))
    points.append(p2 + (v_axial * d_ax) - (v_perp * d_perp))
    points.append(p2 + (v_axial * d_ax) + (v_perp * d_perp))
    points.append(p1 - (v_axial * d_ax) + (v_perp * d_perp))

    # Concatenate
    points = [p.data for p in points]
    points = Vxy(np.concatenate(points, axis=1))

    return LoopXY.from_vertices(points)


def detect_blobs(image: np.ndarray, params: cv.SimpleBlobDetector_Params) -> Vxy:
    """Detects blobs in image. Blobs are defined as local dark regions in
    neighboring light background.

    Parameters
    ----------
    image : np.ndarray
        Input image, uint8
    params : cv.SimpleBlobDetector_Params
        Blob parameters

    Returns
    -------
    Vxy
        Centroids of blobs
    """
    keypoints = _detect_blobs_keypoints(image, params)

    pts = []
    for pt in keypoints:
        pts.append(pt.pt)
    return Vxy(np.array(pts).T)


def detect_blobs_inverse(image: np.ndarray, params: cv.SimpleBlobDetector_Params) -> Vxy:
    """Detect blobs in image. Blobs are defined as local light regions in
    neighboring dark background.

    NOTE: This definition of blobs is the inverse as in `image_processing.detect_blobs()`

    Parameters
    ----------
    image : np.ndarray
        2D input image, single color channel, NxM or NxMx1, uint8
    params : cv.SimpleBlobDetector_Params
        Blob parameters

    Returns
    -------
    Vxy
        Centroids of blobs
    """
    keypoints = _detect_blobs_keypoints(image.max() - image, params)

    pts = []
    for pt in keypoints:
        pts.append(pt.pt)
    return Vxy(np.array(pts).T)


def detect_blobs_annotate(image: np.ndarray, params: cv.SimpleBlobDetector_Params) -> np.ndarray:
    """Detects blobs in image and annotates locations. Blobs are defined as local dark regions in
    neighboring light background.

    Parameters
    ----------
    image : np.ndarray
        2D input image, single color channel, NxM or NxMx1, uint8
    params : cv.SimpleBlobDetector_Params
        Blob parameters

    Returns
    -------
    ndarray
        Annotated image of blobs
    """
    keypoints = _detect_blobs_keypoints(image, params)
    return cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def detect_blobs_inverse_annotate(image: np.ndarray, params: cv.SimpleBlobDetector_Params) -> np.ndarray:
    """Detects blobs in image and annotates locations. Blobs are defined as local light regions in
    neighboring dark background.

    NOTE: This definition of blobs is the inverse as in `image_processing.detect_blobs()`

    Parameters
    ----------
    image : np.ndarray
        2D input image, single color channel, NxM or NxMx1, uint8
    params : cv.SimpleBlobDetector_Params
        Blob parameters

    Returns
    -------
    ndarray
        Annotated image of blobs
    """
    keypoints = _detect_blobs_keypoints(image.max() - image, params)
    return cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def _detect_blobs_keypoints(image: np.ndarray, params: cv.SimpleBlobDetector_Params) -> list[cv.KeyPoint]:
    """
    Detects blobs in image

    Parameters
    ----------
    image : np.ndarray
        2D input image, single color channel, NxM or NxMx1, uint8
    params : cv.SimpleBlobDetector_Params
        Blob parameters

    Returns
    -------
    Vxy
        Centroids of blobs
    """
    # Create a detector
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    return detector.detect(image)


def construct_search_spiral(max_radius: float = 50) -> list[tuple[int, int, float]]:
    """
    Constructs a list of neighboring pixel locations, sorted in order of increasing distance.

    Parameters
    ----------
    max_radius : float
        Maximum distance to search, in units of pixels.  Default 50.

    Returns
    -------
    search_spiral : list[tuple[int, int, float]]
        List of (row, col, distance) tuples, where the "(row, col)" elements indicate
        relative pixel locations, and "distance" is the distance from center of this
        pixel to the center of the listed (row, col) pixel.
        Sorted in order of increasing distance.
    """
    spiral_unsorted = []
    for idx_1 in range(-max_radius, max_radius):
        for idx_2 in range(-max_radius, max_radius):
            r = np.sqrt((idx_1 * idx_1) + (idx_2 * idx_2))
            spiral_unsorted.append((idx_1, idx_2, r))
    return sorted(spiral_unsorted, key=lambda xyr: xyr[2])  # sort by radius


def snap_points_to_nearest_edge(
    surface_points: Vxyz, image_points: Vxy, mask: np.ndarray[bool], search_spiral: list[tuple[int, int, float]]
) -> list[list[tuple[float, float, float], tuple[float, float], tuple[float, float]]]:
    """
    Given a set of (x,y) points describing locations in a binary image, search the
    neighborhood of each point to find the nearest image edge point.

    The maximum search distance is defined by the input search_spiral routine
    construct_search_spiral() above.

    Results are returned as a list of (input_image_point, found_nearest_edge_point)
    pairs, where the input_image_point is one of the input (x,y) points, and the
    found_nearest_edge_point is a (col,row) pair.

    If the routine does not find an edge point for a given input image point
    within the maximum search distance indicated by search_spiral, the the image
    point is skipped and not included in the returned list.

    Image boundaries are ignored, in the sense that an image boundary is not
    viewed as an edge.

    Parameters
    ----------
    surface_points : Vxyz
        Points in 3-d space corresponding to image_points.  Not used in this computation,
        but provided so that returned list will maintain correspondence to 3-d points.
    image_points : Vxy
        Points in image space with (x,y) coordinates to search for nearest edge pixels.
    mask : np.ndarray[bool]
        Binary image to search for edges.
    search_spiral : list[tuple[int, int, float]]
        Sequential list of (row, col, distance) tuples to search, where
        "(row, col)" indicates relative pixel locations, and "distance"
        is the distance to that location.  Sorted in order of increasing
        distance, which implies that search may halt as soon an edge
        pixel is found.

    Returns
    -------
    list[list[tuple[float,float,float],tuple[float,float],tuple[float,float]]]
        List of (3d_point, input_image_point, nearest_edge_point) pairs,
        where
           3d_point corresponds to input_image_point.
           input_image_point is an input (x,y) point.
           found_nearest_edge_point is a (col,row) pair.

        If a given input image point does not have an edge pixel within
        the maximum search distance implied by search spiral, then that
        point is not included in the returned list.
    """
    # Construct search spiral for snapping estimated boundary points to nearest image edge.
    mask_shape = mask.shape
    n_rows = mask_shape[0]
    n_cols = mask_shape[1]
    input_pt_snapped_pt_pair_list = []
    for idx in range(image_points.len()):
        this_vxyz = surface_points[idx]
        this_vxy = image_points[idx]
        this_x = this_vxy.x[0]
        this_y = this_vxy.y[0]
        this_row = round(this_y)
        this_col = round(this_x)
        if (this_row < n_rows) and (this_col < n_cols):
            this_pixel_value = mask[this_row, this_col]
            edge_found = False
            for rcd in search_spiral:
                del_row = rcd[0]
                del_col = rcd[1]
                row = this_row + del_row
                col = this_col + del_col
                if (row < n_rows) and (col < n_cols):
                    loop_pixel_value = mask[row, col]
                    if (this_pixel_value and (not loop_pixel_value)) or ((not this_pixel_value) and loop_pixel_value):
                        edge_found = True
                        break
        if edge_found:
            snap_vxy = Vxy((col, row))
            input_pt_snapped_pt_pair_list.append((this_vxyz, this_vxy, snap_vxy))

    # Return.
    return input_pt_snapped_pt_pair_list
