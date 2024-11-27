#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:17:51 2024

@author: alisha
"""

from sympy.diffgeom import Manifold, Patch
import pystein as pst
import sympy as sp
from sympy import Array, symbols
from pystein import metric, coords, gravity, curvature
from pystein.utilities import tensor_pow as tpow
from sympy import *
init_printing()
a, t, x, y, z = sp.symbols('a t x y z')

hxx, hyy, hzz = sp.symbols('hxx hyy hzz')

hxy, hxz, hyz = sp.symbols('hxy hxz hyz')

M = Manifold('M', dim = 4)
P = Patch('origin', M)
cs = coords.CoordSystem('cartesian', P, ['t','x','y','z'])


dt,dx,dy,dz = cs.base_oneforms()

form = -a**2 * tpow(dt,2)+a**2 * tpow(dx,2)+a**2 * tpow(dy,2)+a**2 * tpow(dz,2)

g1 = metric.Metric(twoform=form)

matrix = Array([[-a**2, 0, 0, 0],[0, a**2+a**2*hxx, a**2*hxy, a**2*hxz],
                [0,a**2*hxy, a**2+a**2*hyy, a**2*hyz],[0, a**2*hxz, a**2*hyz, a**2+a**2*hzz]])

g2 = metric.Metric(matrix=matrix, coord_system=cs)

christoffels, riemanns, riccis = curvature.compute_components(metric=g2)