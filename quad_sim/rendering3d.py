"""
3D rendering framework
"""
from __future__ import division
from copy import deepcopy
import os
import six
import sys
import itertools
import noise
import ctypes

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

# from gym.utils import reraise
from gym import error
import matplotlib.pyplot as plt

try:
    import pyglet
    pyglet.options['debug_gl'] = False
except ImportError as e:
        raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')
    # reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
        raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')
    # reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

# TODO can we get some of this from Pyglet?
class FBOTarget(object):
    def __init__(self, width, height):

        shape = (width, height, 3)
        self.shape = shape

        self.fbo = GLuint(0)
        glGenFramebuffers(1, ctypes.byref(self.fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # renderbuffer for depth
        self.depth = GLuint(0)
        glGenRenderbuffers(1, ctypes.byref(self.depth))
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, *shape)
        # ??? (from songho.ca/opengl/gl_fbo.html)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER, self.depth)

        # texture for RGB
        self.tex = GLuint(0)
        glGenTextures(1, ctypes.byref(self.tex))
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, shape[0], shape[1], 0,
            GL_RGB, GL_UNSIGNED_BYTE, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self.tex, 0)

        # test - ok to comment out?
        draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)
        glDrawBuffers(1, draw_buffers)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        self.fb_array = np.zeros(shape, dtype=np.uint8)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)
        glDrawBuffers(1, draw_buffers)
        glViewport(0, 0, *self.shape[:2])

    def finish(self):
        glReadPixels(0, 0, self.shape[1], self.shape[0],
            GL_RGB, GL_UNSIGNED_BYTE, self.fb_array.ctypes.data)

    def read(self):
        return self.fb_array


class WindowTarget(object):
    def __init__(self, width, height, display=None, resizable=True):

        config=Config(double_buffer=True, depth_size=16)
        display = get_display(display)
        # vsync is set to false to speed up FBO-only renders, we enable before draw
        self.window = pyglet.window.Window(display=display,
            width=width, height=height, resizable=resizable,
            visible=True, vsync=False, config=config
        )
        self.window.on_close = self.close
        self.shape = (width, height, 3)
        def on_resize(w, h):
            self.shape = (w, h, 3)
        if resizable:
            self.window.on_resize = on_resize

    def close(self):
        self.window.close()

    def bind(self):
        self.window.switch_to()
        self.window.set_vsync(True)
        self.window.dispatch_events()
        glViewport(0, 0, self.window.width, self.window.height)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def finish(self):
        self.window.flip()
        self.window.set_vsync(False)


class Camera(object):
    def __init__(self, fov):
        self.fov = fov
        self.lookat = None

    def look_at(self, eye, target, up):
        self.lookat = (eye, target, up)

    # TODO other ways to set the view matrix

    # private
    def _matrix(self, shape):
        aspect = float(shape[0]) / shape[1]
        znear = 0.1
        zfar = 100.0
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, aspect, znear, zfar)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # will make sense once more than one way of setting view matrix
        assert sum([x is not None for x in (self.lookat,)]) < 2

        if self.lookat is not None:
            eye, target, up = (list(x) for x in self.lookat)
            gluLookAt(*(eye + target + up))


# TODO we can add user-controlled lighting, etc. to this
class Scene(object):
    def __init__(self, batches, bgcolor=(0,0,0)):
        self.batches = batches
        self.bgcolor = bgcolor

        # [-1] == 0 means it's a directional light
        self.lights = [np.array([np.cos(t), np.sin(t), 0.0, 0.0])
            for t in 0.2 + np.linspace(0, 2*np.pi, 4)[:-1]]

    # call only once GL context is ready
    def initialize(self):
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHTING)

        #glFogi(GL_FOG_MODE, GL_LINEAR)
        #glFogf(GL_FOG_START, 20.0) # Fog Start Depth
        #glFogf(GL_FOG_END, 100.0) # Fog End Depth
        #glEnable(GL_FOG)

        amb, diff, spec = (1.0 / len(self.lights)) * np.array([0.4, 1.2, 0.5])
        for i, light in enumerate(self.lights):
            # TODO fix lights in world space instead of camera space
            glLightfv(GL_LIGHT0 + i, GL_POSITION, (GLfloat * 4)(*light))
            glLightfv(GL_LIGHT0 + i, GL_AMBIENT, (GLfloat * 4)(amb, amb, amb, 1))
            glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, (GLfloat * 4)(diff, diff, diff, 1))
            glLightfv(GL_LIGHT0 + i, GL_SPECULAR, (GLfloat * 4)(spec, spec, spec, 1))
            glEnable(GL_LIGHT0 + i)


def draw(scene, camera, target):

    target.bind() # sets viewport

    r, g, b = scene.bgcolor
    glClearColor(r, g, b, 1.0)
    glFrontFace(GL_CCW)
    glCullFace(GL_BACK)
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    camera._matrix(target.shape)
    #view = (GLfloat * 16)()
    #glGetFloatv(GL_MODELVIEW_MATRIX, view)
    #view = np.array(view).reshape((4,4)).T

    for batch in scene.batches:
        batch.draw()

    target.finish()


class SceneNode(object):
    def _build_children(self, batch):
        # hack - should go somewhere else
        if not isinstance(self.children, type([])):
            self.children = [self.children]
        for c in self.children:
            c.build(batch, self.pyg_grp)

    # default impl
    def collide_sphere(self, x, radius):
        return any(c.collide_sphere(x, radius) for c in self.children)

class World(SceneNode):
    def __init__(self, children):
        self.children = children
        self.pyg_grp = None

    def build(self, batch):
        self._build_children(batch)

class Transform(SceneNode):
    def __init__(self, transform, children):
        self.t = transform
        self.mat_inv = np.linalg.inv(transform)
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = _PygTransform(self.t, parent=parent)
        self._build_children(batch)
        return self.pyg_grp

    def set_transform(self, t):
        self.pyg_grp.set_matrix(t)
        self.mat_inv = np.linalg.inv(t)

    def set_transform_nocollide(self, t):
        self.pyg_grp.set_matrix(t)

    def collide_sphere(self, x, radius):
        xh = [x[0], x[1], x[2], 1]
        xlocal = np.matmul(self.mat_inv, xh)[:3]
        rlocal = radius * self.mat_inv[0,0]
        return any(c.collide_sphere(xlocal, rlocal) for c in self.children)

class BackToFront(SceneNode):
    def __init__(self, children):
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = pyglet.graphics.Group(parent=parent)
        for i, c in enumerate(self.children):
            ordering = pyglet.graphics.OrderedGroup(i, parent=self.pyg_grp)
            c.build(batch, ordering)
        return self.pyg_grp

class Color(SceneNode):
    def __init__(self, color, children):
        self.color = color
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = _PygColor(self.color, parent=parent)
        self._build_children(batch)
        return self.pyg_grp

    def set_rgb(self, r, g, b):
        self.pyg_grp.set_rgb(r, g, b)

def transform_and_color(transform, color, children):
    return Transform(transform, Color(color, children))

TEX_CHECKER = 0
TEX_XOR = 1
TEX_NOISE_GAUSSIAN = 2
TEX_NOISE_PERLIN = 3
TEX_OILY = 4
TEX_VORONOI = 5

def random_textype():
    return np.random.randint(TEX_VORONOI + 1)

class ProceduralTexture(SceneNode):
    def __init__(self, style, scale, children):
        self.children = children
        # linear is default, those w/ nearest must overwrite
        self.mag_filter = GL_LINEAR
        if style == TEX_CHECKER:
            image = np.zeros((256, 256))
            image[:128,:128] = 1.0
            image[128:,128:] = 1.0
            self.mag_filter = GL_NEAREST
        elif style == TEX_XOR:
            x, y = np.meshgrid(range(256), range(256))
            image = np.float32(np.bitwise_xor(np.uint8(x), np.uint8(y)))
            self.mag_filter = GL_NEAREST
        elif style == TEX_NOISE_GAUSSIAN:
            nz = np.random.normal(size=(256,256))
            image = np.clip(nz, -3, 3)
        elif style == TEX_NOISE_PERLIN:
            t = np.linspace(0, 1, 256)
            nzfun = lambda x, y: noise.pnoise2(x, y,
                octaves=10, persistence=0.8, repeatx=1, repeaty=1)
            image = np.vectorize(nzfun)(*np.meshgrid(t, t))
        elif style == TEX_OILY:
            # from upvector.com "Intro to Procedural Textures"
            t = np.linspace(0, 4, 256)
            nzfun = lambda x, y: noise.snoise2(x, y,
                octaves=10, persistence=0.45, repeatx=4, repeaty=4)
            nz = np.vectorize(nzfun)(*np.meshgrid(t, t))

            t = np.linspace(0, 20*np.pi, 257)[:-1]
            x, y = np.meshgrid(t, t)
            image = np.sin(x + 8*nz)
        elif style == TEX_VORONOI:
            npts = 64
            points = np.random.uniform(size=(npts, 2))
            # make it tile
            shifts = itertools.product([-1, 0, 1], [-1, 0, 1])
            points = np.vstack([points + shift for shift in shifts])
            unlikely = np.any(np.logical_or(points < -0.25, points > 1.25), axis=1)
            points = np.delete(points, np.where(unlikely), axis=0)
            a = np.full((256, 256), np.inf)
            t = np.linspace(0, 1, 256)
            x, y = np.meshgrid(t, t)
            for p in points:
                dist2 = (x - p[0])**2 + (y - p[1])**2
                a = np.minimum(a, dist2)
            image = np.sqrt(a)
        else:
            raise KeyError("style does not exist")

        low, high = 255.0 * scale[0], 255.0 * scale[1]
        _scale_to_inplace(image, low, high)
        self.tex = _np2tex(image)

    def build(self, batch, parent):
        self.pyg_grp = _PygTexture(tex=self.tex,
            mag_filter=self.mag_filter, parent=parent)
        self._build_children(batch)
        return self.pyg_grp


#
# these functions return 4x4 rotation matrix suitable to construct Transform
# or to mutate Transform via set_matrix
#
def scale(s):
    return np.diag([s, s, s, 1.0])

def translate(x):
    r = np.eye(4)
    r[:3,3] = x
    return r

def trans_and_rot(t, r):
    m = np.eye(4)
    m[:3,:3] = r
    m[:3,3] = t
    return m

def rotz(theta):
    r = np.eye(4)
    r[:2,:2] = _rot2d(theta)
    return r

def roty(theta):
    r = np.eye(4)
    r2d = _rot2d(theta)
    r[[0,0,2,2],[0,2,0,2]] = _rot2d(theta).flatten()
    return r

def rotx(theta):
    r = np.eye(4)
    r[1:3,1:3] = _rot2d(theta)
    return r

class _PygTransform(pyglet.graphics.Group):
    def __init__(self, transform=np.eye(4), parent=None):
        super().__init__(parent)
        self.set_matrix(transform)

    def set_matrix(self, transform):
        assert transform.shape == (4, 4)
        assert np.all(transform[3,:] == [0, 0, 0, 1])
        self.matrix_raw = (GLfloat * 16)(*transform.T.flatten())

    def set_state(self):
        glPushMatrix()
        glMultMatrixf(self.matrix_raw)

    def unset_state(self):
        glPopMatrix()

class _PygColor(pyglet.graphics.Group):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        if len(color) == 3:
            self.set_rgb(*color)
        else:
            self.set_rgba(*color)

    def set_rgb(self, r, g, b):
        self.set_rgba(r, g, b, 1.0)

    def set_rgba(self, r, g, b, a):
        self.dcolor = (GLfloat * 4)(r, g, b, a)
        spec_whiteness = 0.8
        r, g, b = (1.0 - spec_whiteness) * np.array([r, g, b]) + spec_whiteness
        self.scolor = (GLfloat * 4)(r, g, b, a)

    def set_state(self):
        if self.dcolor[-1] < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.dcolor)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.scolor)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (GLfloat)(8.0))

    def unset_state(self):
        if self.dcolor[-1] < 1.0:
            glDisable(GL_BLEND)

class _PygTexture(pyglet.graphics.Group):
    def __init__(self, tex, mag_filter, parent=None):
        super().__init__(parent=parent)

        self.tex = tex
        glBindTexture(GL_TEXTURE_2D, self.tex.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # anisotropic texturing helps a lot with checkerboard floors
        anisotropy = (GLfloat)()
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)

    def set_state(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(1,1,1,1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(1,1,1,1))
        glEnable(self.tex.target)
        glBindTexture(self.tex.target, self.tex.id)

    def unset_state(self):
        glDisable(self.tex.target)


class _PygAlphaBlending(pyglet.graphics.Group):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def set_state(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self):
        glDisable(GL_BLEND)

Batch = pyglet.graphics.Batch

# we only implement collision detection between primitives and spheres.
# the world-coordinate sphere is transformed according to the scene graph
# into the primitive's canonical coordinate system.
# this simplifies the math a lot, although it might be less efficient
# than directly testing the sphere against the transformed primitive
# in world coordinates.
class SphereCollision(object):
    def __init__(self, radius):
        self.radius = radius
    def collide_sphere(self, x, radius):
        c = np.sum(x ** 2) < (self.radius + radius) ** 2
        if c: print("collided with sphere")
        return c

class AxisBoxCollision(object):
    def __init__(self, corner0, corner1):
        self.corner0, self.corner1 = corner0, corner1
    def collide_sphere(self, x, radius):
        nearest = np.maximum(self.corner0, np.minimum(x, self.corner1))
        c = np.sum((x - nearest)**2) < radius**2
        if c: print("collided with box")
        return c

class CapsuleCollision(object):
    def __init__(self, radius, height):
        self.radius, self.height = radius, height
    def collide_sphere(self, x, radius):
        z = min(max(0, x[2]), self.height)
        nearest = [0, 0, z]
        c = np.sum((x - nearest)**2) < (self.radius + radius)**2
        if c: print("collided with capsule")
        return c

#
# these are the 3d primitives that can be added to a pyglet.graphics.Batch.
# construct them with the shape functions below.
#
class BatchElement(SceneNode):
    def build(self, batch, parent):
        self.batch_args[2] = parent
        batch.add(*self.batch_args)

    def collide_sphere(self, x, radius):
        if self.collider is not None:
            return self.collider.collide_sphere(x, radius)
        else:
            return False

class Mesh(BatchElement):
    def __init__(self, verts, normals=None, st=None, collider=None):
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            raise ValueError('verts must be an N x 3 NumPy array')

        N = verts.shape[0]
        assert int(N) % 3 == 0

        if st is not None:
            assert st.shape == (N, 2)

        if normals is None:
            # compute normals implied by triangle faces
            normals = deepcopy(verts)

            for i in range(0, N, 3):
                v0, v1, v2 = verts[i:(i+3),:]
                d0, d1 = (v1 - v0), (v2 - v1)
                n = _normalize(np.cross(d0, d1))
                normals[i:(i+3),:] = n

        self.batch_args = [N, pyglet.gl.GL_TRIANGLES, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten())),
        ]
        if st is not None:
            self.batch_args.append(('t2f/static', list(st.flatten())))
        self.collider = collider

class TriStrip(BatchElement):
    def __init__(self, verts, normals, collider=None):
        N, dim = verts.shape
        assert dim == 3
        assert normals.shape == verts.shape

        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_STRIP, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]
        self.collider = collider

class TriFan(BatchElement):
    def __init__(self, verts, normals, collider=None):
        N, dim = verts.shape
        assert dim == 3
        assert normals.shape == verts.shape

        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_FAN, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]
        self.collider = collider

# a box centered on the origin
def box(x, y, z):
    corner1 = np.array([x,y,z]) / 2
    corner0 = -corner1
    v = box_mesh(x, y, z)
    collider = AxisBoxCollision(corner0, corner1)
    return Mesh(v, collider=collider)

# cylinder sitting on xy plane pointing +z
def cylinder(radius, height, sections):
    v, n = cylinder_strip(radius, height, sections)
    collider = CapsuleCollision(radius, height)
    return TriStrip(v, n, collider=collider)

# cone sitting on xy plane pointing +z
def cone(radius, height, sections):
    # TODO collision detectoin
    v, n = cone_strip(radius, height, sections)
    return TriStrip(v, n)

# arrow sitting on xy plane pointing +z
def arrow(radius, height, sections):
    v, n = arrow_strip(radius, height, sections)
    return TriStrip(v, n)

# sphere centered on origin, n tris will be about TODO * facets
def sphere(radius, facets):
    v, n = sphere_strip(radius, facets)
    collider = SphereCollision(radius)
    return TriStrip(v, n, collider=collider)

# square in xy plane centered on origin
# dim: (w, h)
# srange, trange: desired min/max (s, t) tex coords
def rect(dim, srange=(0,1), trange=(0,1)):
    v = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0],
        [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    v = np.matmul(v, np.diag([dim[0] / 2.0, dim[1] / 2.0, 0]))
    n = _withz(0 * v, 1)
    s0, s1 = srange
    t0, t1 = trange
    st = np.array([
        [s1, t1], [s0, t1], [s1, t0],
        [s0, t1], [s0, t0], [s1, t0]])
    return Mesh(v, n, st)

def circle(radius, facets):
    v, n = circle_fan(radius, facets)
    return TriFan(v, n)

#
# low-level primitive builders. return vertex/normal/texcoord arrays.
# good if you want to apply transforms directly to the points, etc.
#

# box centered on origin with given dimensions.
# no normals, but Mesh ctor will estimate them perfectly
def box_mesh(x, y, z):
    vtop = np.array([[x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z]])
    vbottom = deepcopy(vtop)
    vbottom[:,2] = -vbottom[:,2]
    v = 0.5 * np.concatenate([vtop, vbottom], axis=0)
    t = np.array([[1, 3, 2,], [1, 4, 3,], [1, 2, 5,], [2, 6, 5,], [2, 3, 6,], [3, 7, 6,], [3, 4, 8,], [3, 8, 7,], [4, 1, 8,], [1, 5, 8,], [5, 6, 7,], [5, 7, 8,]]) - 1
    t = t.flatten()
    v = v[t,:]
    return v

# circle in the x-y plane
def circle_fan(radius, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    v = np.hstack([x, y, 0*t])
    v = np.vstack([[0, 0, 0], v])
    n = _withz(0 * v, 1)
    return v, n

# cylinder sitting on the x-y plane
def cylinder_strip(radius, height, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    base = np.hstack([x, y, 0*t])
    top = np.hstack([x, y, height + 0*t])
    strip_sides = _to_strip(np.hstack([base[:,None,:], top[:,None,:]]))
    normals_sides = _withz(strip_sides / radius, 0)

    def make_cap(circle, normal_z):
        height = circle[0,2]
        center = _withz(0 * circle, height)
        if normal_z > 0:
            strip = _to_strip(np.hstack([circle[:,None,:], center[:,None,:]]))
        else:
            strip = _to_strip(np.hstack([center[:,None,:], circle[:,None,:]]))
        normals = _withz(0 * strip, normal_z)
        return strip, normals

    vbase, nbase = make_cap(base, -1)
    vtop, ntop = make_cap(top, 1)
    return (
        np.vstack([strip_sides, vbase, vtop]),
        np.vstack([normals_sides, nbase, ntop]))

# cone sitting on the x-y plane
def cone_strip(radius, height, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    base = np.hstack([x, y, 0*t])

    top = _withz(0 * base, height)
    vside = _to_strip(np.hstack([base[:,None,:], top[:,None,:]]))
    base_tangent = np.cross(_npa(0, 0, 1), base)
    top_to_base = base - top
    normals = _normalize(np.cross(top_to_base, base_tangent))
    nside = _to_strip(np.hstack([normals[:,None,:], normals[:,None,:]]))

    base_ctr = 0 * base
    vbase = _to_strip(np.hstack([base_ctr[:,None,:], base[:,None,:]]))
    nbase = _withz(0 * vbase, -1)

    return np.vstack([vside, vbase]), np.vstack([nside, nbase])

# sphere centered on origin
def sphere_strip(radius, resolution):
    t = np.linspace(-1, 1, resolution)
    u, v = np.meshgrid(t, t)
    vtx = []
    panel = np.zeros((resolution, resolution, 3))
    inds = list(range(3))
    for i in range(3):
        panel[:,:,inds[0]] = u
        panel[:,:,inds[1]] = v
        panel[:,:,inds[2]] = 1
        norms = np.linalg.norm(panel, axis=2)
        panel = panel / norms[:,:,None]
        for _ in range(2):
            for j in range(resolution - 1):
                strip = deepcopy(panel[[j,j+1],:,:].transpose([1,0,2]).reshape((-1,3)))
                degen0 = deepcopy(strip[0,:])
                degen1 = deepcopy(strip[-1,:])
                vtx.extend([degen0, strip, degen1])
            panel *= -1
            panel = np.flip(panel, axis=1)
        inds = [inds[-1]] + inds[:-1]

    n = np.vstack(vtx)
    v = radius * n
    return v, n

# arrow sitting on x-y plane
def arrow_strip(radius, height, facets):
    cyl_r = radius
    cyl_h = 0.75 * height
    cone_h = height - cyl_h
    cone_half_angle = np.radians(30)
    cone_r = cone_h * np.tan(cone_half_angle)
    vcyl, ncyl = cylinder_strip(cyl_r, cyl_h, facets)
    vcone, ncone = cone_strip(cone_r, cone_h, facets)
    vcone[:,2] += cyl_h
    v = np.vstack([vcyl, vcone])
    n = np.vstack([ncyl, ncone])
    return v, n


#
# private helper functions, not part of API
#
def _npa(*args):
    return np.array(args)

def _normalize(x):
    if len(x.shape) == 1:
        return x / np.linalg.norm(x)
    elif len(x.shape) == 2:
        return x / np.linalg.norm(x, axis=1)[:,None]
    else:
        assert False

def _withz(a, z):
    b = 0 + a
    b[:,2] = z
    return b

def _rot2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

# add degenerate tris, convert from N x 2 x 3 to 2N+2 x 3
def _to_strip(strip):
    s0 = strip[0,0,:]
    s1 = strip[-1,-1,:]
    return np.vstack([s0, np.reshape(strip, (-1, 3)), s1])

def _scale_to_inplace(a, min1, max1):
    id0 = id(a)
    min0 = np.min(a.flatten())
    max0 = np.max(a.flatten())
    scl = (max1 - min1) / (max0 - min0)
    shift = - (scl * min0) + min1
    a *= scl
    a += shift
    assert id(a) == id0

def _np2tex(a):
    # TODO color
    w, h = a.shape
    b = np.uint8(a).tobytes()
    assert len(b) == w * h
    img = pyglet.image.ImageData(w, h, "L", b)
    return img.get_texture()


