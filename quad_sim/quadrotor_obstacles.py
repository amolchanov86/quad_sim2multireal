import numpy as np
from numpy.linalg import norm
from gym_art.quadrotor.quad_utils import *
import gym_art.quadrotor.rendering3d as r3d

# determine where to put the obstacles such that no two obstacles intersect
# and compute the list of obstacles to collision check at each 2d tile.
def _place_obstacles(np_random, N, box, radius_range, our_radius, tries=5):

    t = np.linspace(0, box, TILES+1)[:-1]
    scale = box / float(TILES)
    x, y = np.meshgrid(t, t)
    pts = np.zeros((N, 2))
    dist = x + np.inf

    radii = np_random.uniform(*radius_range, size=N)
    radii = np.sort(radii)[::-1]
    test_list = [[] for i in range(TILES**2)]

    for i in range(N):
        rad = radii[i]
        ok = np.where(dist.flat > rad)[0]
        if len(ok) == 0:
            if tries == 1:
                print("Warning: only able to place {}/{} obstacles. "
                    "Increase box, decrease radius, or decrease N.")
                return pts[:i,:], radii[:i]
            else:
                return _place_obstacles(N, box, radius_range, tries-1)
        pt = np.unravel_index(np_random.choice(ok), dist.shape)
        pt = scale * np.array(pt)
        d = np.sqrt((x - pt[1])**2 + (y - pt[0])**2) - rad
        # big slop factor for tile size, off-by-one errors, etc
        for ind1d in np.where(d.flat <= 2*our_radius + scale)[0]:
            test_list[ind1d].append(i)
        dist = np.minimum(dist, d)
        pts[i,:] = pt - box/2.0

    # very coarse to allow for binning bugs
    test_list = np.array(test_list).reshape((TILES, TILES))
    #amt_free = sum(len(a) == 0 for a in test_list.flat) / float(test_list.size)
    #print(amt_free * 100, "pct free space")
    return pts, radii, test_list


# generate N obstacles w/ randomized primitive, size, color, TODO texture
# arena: boundaries of world in xy plane
# our_radius: quadrotor's radius
def _random_obstacles(np_random, N, arena, our_radius):
    arena = float(arena)
    # all primitives should be tightly bound by unit circle in xy plane
    boxside = np.sqrt(2)
    box = r3d.box(boxside, boxside, boxside)
    sphere = r3d.sphere(radius=1.0, facets=16)
    cylinder = r3d.cylinder(radius=1.0, height=2.0, sections=32)
    # TODO cone-sphere collision
    #cone = r3d.cone(radius=0.5, height=1.0, sections=32)
    primitives = [box, sphere, cylinder]

    bodies = []
    max_radius = 2.0
    positions, radii, test_list = _place_obstacles(
        np_random, N, arena, (0.5, max_radius), our_radius)
    for i in range(N):
        primitive = np_random.choice(primitives)
        tex_type = r3d.random_textype()
        tex_dark = 0.5 * np_random.uniform()
        tex_light = 0.5 * np_random.uniform() + 0.5
        color = 0.5 * np_random.uniform(size=3)
        heightscl = np.random.uniform(0.5, 2.0)
        height = heightscl * 2.0 * radii[i]
        z = (0 if primitive is cylinder else
            (height/2.0 if primitive is sphere else
            (height*boxside/4.0 if primitive is box
            else np.nan)))
        translation = np.append(positions[i,:], z)
        matrix = np.matmul(r3d.translate(translation), r3d.scale(radii[i]))
        matrix = np.matmul(matrix, np.diag([1, 1, heightscl, 1]))
        body = r3d.Transform(matrix,
            #r3d.ProceduralTexture(tex_type, (tex_dark, tex_light), primitive))
                r3d.Color(color, primitive))
        bodies.append(body)

    return ObstacleMap(arena, bodies, test_list)


# main class for non-visual aspects of the obstacle map.
class ObstacleMap(object):
    def __init__(self, box, bodies, test_lists):
        self.box = box
        self.bodies = bodies
        self.test = test_lists

    def detect_collision(self, dynamics):
        pos = dynamics.pos
        if pos[2] <= dynamics.arm:
            print("collided with terrain")
            return True
        r, c = self.coord2tile(*dynamics.pos[:2])
        if r < 0 or c < 0 or r >= TILES or c >= TILES:
            print("collided with wall")
            return True
        if self.test is not None:
            radius = dynamics.arm + 0.1
            return any(self.bodies[k].collide_sphere(pos, radius)
                for k in self.test[r,c])
        return False

    def sample_start(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((pad, pad + band), np_random)

    def sample_goal(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((-(pad + band), -pad), np_random)

    def sample_freespace(self, rowrange, np_random):
        rfree, cfree = np.where(np.vectorize(lambda t: len(t) == 0)(
            self.test[rowrange[0]:rowrange[1],:]))
        choice = np_random.choice(len(rfree))
        r, c = rfree[choice], cfree[choice]
        r += rowrange[0]
        x, y = self.tile2coord(r, c)
        z = np_random.uniform(1.0, 3.0)
        return np.array([x, y, z])

    def tile2coord(self, r, c):
        #TODO consider moving origin to corner of world
        scale = self.box / float(TILES)
        return scale * np.array([r,c]) - self.box / 2.0

    def coord2tile(self, x, y):
        scale = float(TILES) / self.box
        return np.int32(scale * (np.array([x,y]) + self.box / 2.0))