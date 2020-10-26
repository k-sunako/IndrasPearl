#!/usr/bin/env python

import math
import cairo

# WIDTH, HEIGHT = 256, 256
WIDTH, HEIGHT = 1024, 1024

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
ctx.scale(WIDTH, HEIGHT)  # Normalizing the canvas


# Backgound
pat = cairo.SolidPattern(1.0, 1.0, 1.0, alpha=0.0)
ctx.rectangle(0.0, 0.0, 1.0, 1.0)
ctx.set_source(pat)
ctx.fill()

# Arc(cx, cy, radius, start_angle, stop_angle)
ctx.set_source_rgb(1.0, 0.0, 0.0)
ctx.arc(0.5, 0.5, 0.4, 0, 2 * math.pi)
ctx.fill()

ctx.set_source_rgb(0.0, 1.0, 0.0)
ctx.arc(0.4, 0.4, 0.2, 0, 2 * math.pi)
ctx.fill()

surface.write_to_png("two-circle.png")  # Output to PNG
