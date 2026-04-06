bl_info = {
    "name": "UV Texture Link Rescaler",
    "author": "Codex",
    "version": (1, 9, 0),
    "blender": (3, 6, 0),
    "location": "UV Editor > Sidebar > UV Tex Linbl_info = {
    "name": "UV Texture Link Rescaler",
    "author": "Codex",
    "version": (1, 15, 0),
    "blender": (3, 6, 0),
    "location": "UV Editor > Sidebar > UV Tex Link",
    "description": "Sample UV/texture bounds and rescale texture pixels to match moved UV islands",
    "category": "UV",
}

import math
from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import bmesh
import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup


# In-memory pixel snapshots keyed by scene pointer.
_PIXEL_BUFFERS: Dict[int, Dict[str, object]] = {}
_EPSILON = 1e-8
_GEOM_EPSILON = 1e-14
_UV_TOLERANCE = 1e-6
_TRI_BIN_SIZE = 32


Bounds = Tuple[float, float, float, float]
UVCoord = Tuple[float, float]
LoopRef = Tuple[int, int]
TriangleUV = Tuple[float, float, float, float, float, float]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_mesh_edit_mode(context: bpy.types.Context) -> bool:
    obj = context.active_object
    return bool(obj and obj.type == "MESH" and context.mode == "EDIT_MESH")


def _loop_selected(
    context: bpy.types.Context,
    face: bmesh.types.BMFace,
    loop: bmesh.types.BMLoop,
    luv: bmesh.types.BMLoopUV,
) -> bool:
    if not context.scene.tool_settings.use_uv_select_sync:
        return luv.select

    select_mode = context.scene.tool_settings.mesh_select_mode
    if select_mode[2]:
        return face.select
    if select_mode[1]:
        return loop.edge.select
    return loop.vert.select


def _normalize_bounds(bounds: Bounds) -> Bounds:
    min_u, min_v, max_u, max_v = bounds
    return (min(min_u, max_u), min(min_v, max_v), max(min_u, max_u), max(min_v, max_v))


def _format_bounds(bounds: Bounds) -> str:
    min_u, min_v, max_u, max_v = bounds
    return f"({min_u:.4f}, {min_v:.4f}) -> ({max_u:.4f}, {max_v:.4f})"


def _debug_compact(details: Dict[str, object]) -> str:
    ordered = []
    for key in sorted(details.keys()):
        ordered.append(f"{key}={details[key]}")
    return " || ".join(ordered)


def _recommended_bleed_px_for_resolution(size_px: int) -> int:
    if size_px >= 8192:
        return 32
    if size_px >= 4096:
        return 16
    if size_px >= 2048:
        return 8
    if size_px >= 1024:
        return 4
    if size_px >= 512:
        return 2
    return 1


def _bounds_almost_equal(a: Bounds, b: Bounds, eps: float = 1e-7) -> bool:
    a0, a1, a2, a3 = _normalize_bounds(a)
    b0, b1, b2, b3 = _normalize_bounds(b)
    return (
        abs(a0 - b0) <= eps
        and abs(a1 - b1) <= eps
        and abs(a2 - b2) <= eps
        and abs(a3 - b3) <= eps
    )


def _union_bounds(bounds_list: Sequence[Bounds]) -> Optional[Bounds]:
    if not bounds_list:
        return None

    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    for bounds in bounds_list:
        b_min_u, b_min_v, b_max_u, b_max_v = _normalize_bounds(bounds)
        min_u = min(min_u, b_min_u)
        min_v = min(min_v, b_min_v)
        max_u = max(max_u, b_max_u)
        max_v = max(max_v, b_max_v)

    return (min_u, min_v, max_u, max_v)


def _uv_equal(a: UVCoord, b: UVCoord, tolerance: float = _UV_TOLERANCE) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _edge_uv_pair(
    face: bmesh.types.BMFace,
    edge: bmesh.types.BMEdge,
    uv_layer: object,
) -> Optional[Tuple[UVCoord, UVCoord]]:
    for loop in face.loops:
        if loop.edge == edge:
            uv_a = loop[uv_layer].uv
            uv_b = loop.link_loop_next[uv_layer].uv
            return (float(uv_a.x), float(uv_a.y)), (float(uv_b.x), float(uv_b.y))
    return None


def _faces_uv_connected(
    face_a: bmesh.types.BMFace,
    face_b: bmesh.types.BMFace,
    edge: bmesh.types.BMEdge,
    uv_layer: object,
) -> bool:
    pair_a = _edge_uv_pair(face_a, edge, uv_layer)
    pair_b = _edge_uv_pair(face_b, edge, uv_layer)
    if pair_a is None or pair_b is None:
        return False

    a0, a1 = pair_a
    b0, b1 = pair_b
    return (_uv_equal(a0, b0) and _uv_equal(a1, b1)) or (_uv_equal(a0, b1) and _uv_equal(a1, b0))


def _bounds_from_loop_refs_bmesh(
    bm: bmesh.types.BMesh,
    uv_layer: object,
    loop_refs: Sequence[LoopRef],
) -> Optional[Bounds]:
    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")
    found = False

    for face_index, loop_slot in loop_refs:
        if face_index < 0 or face_index >= len(bm.faces):
            return None

        face = bm.faces[face_index]
        if loop_slot < 0 or loop_slot >= len(face.loops):
            return None

        uv = face.loops[loop_slot][uv_layer].uv
        found = True
        min_u = min(min_u, float(uv.x))
        min_v = min(min_v, float(uv.y))
        max_u = max(max_u, float(uv.x))
        max_v = max(max_v, float(uv.y))

    if not found:
        return None

    return (min_u, min_v, max_u, max_v)


def _face_uv_triangles_from_face_indices(
    bm: bmesh.types.BMesh,
    uv_layer: object,
    face_indices: Sequence[int],
) -> Optional[list]:
    triangles = []

    for face_index in face_indices:
        if face_index < 0 or face_index >= len(bm.faces):
            return None
        face = bm.faces[face_index]
        if len(face.loops) < 3:
            continue

        uv0 = face.loops[0][uv_layer].uv
        u0 = float(uv0.x)
        v0 = float(uv0.y)

        for i in range(1, len(face.loops) - 1):
            uv1 = face.loops[i][uv_layer].uv
            uv2 = face.loops[i + 1][uv_layer].uv
            triangles.append(
                (
                    u0,
                    v0,
                    float(uv1.x),
                    float(uv1.y),
                    float(uv2.x),
                    float(uv2.y),
                )
            )

    return triangles


def _point_in_triangle(u: float, v: float, tri: TriangleUV, eps: float = 1e-9) -> bool:
    weights = _barycentric_weights(u, v, tri)
    if weights is None:
        return False
    w0, w1, w2 = weights
    return (
        w0 >= -eps
        and w1 >= -eps
        and w2 >= -eps
        and w0 <= 1.0 + eps
        and w1 <= 1.0 + eps
        and w2 <= 1.0 + eps
    )


def _point_in_triangles(u: float, v: float, triangles: Sequence[TriangleUV]) -> bool:
    for tri in triangles:
        if _point_in_triangle(u, v, tri):
            return True
    return False


def _triangle_bounds(tri: TriangleUV) -> Bounds:
    u0, v0, u1, v1, u2, v2 = tri
    return (
        min(u0, u1, u2),
        min(v0, v1, v2),
        max(u0, u1, u2),
        max(v0, v1, v2),
    )


def _triangles_bounds(triangles: Sequence[TriangleUV]) -> Optional[Bounds]:
    if not triangles:
        return None

    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    for tri in triangles:
        b_min_u, b_min_v, b_max_u, b_max_v = _triangle_bounds(tri)
        min_u = min(min_u, b_min_u)
        min_v = min(min_v, b_min_v)
        max_u = max(max_u, b_max_u)
        max_v = max(max_v, b_max_v)

    return (min_u, min_v, max_u, max_v)


def _triangles_max_delta(a: Sequence[TriangleUV], b: Sequence[TriangleUV]) -> float:
    if len(a) != len(b):
        return float("inf")

    max_delta = 0.0
    for tri_a, tri_b in zip(a, b):
        for va, vb in zip(tri_a, tri_b):
            delta = abs(float(va) - float(vb))
            if delta > max_delta:
                max_delta = delta
    return max_delta


def _barycentric_weights(u: float, v: float, tri: TriangleUV) -> Optional[Tuple[float, float, float]]:
    u0, v0, u1, v1, u2, v2 = tri
    denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    if abs(denom) <= _GEOM_EPSILON:
        return None

    w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
    w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
    w2 = 1.0 - w0 - w1
    return (w0, w1, w2)


def _remap_uv_between_bounds(
    u: float,
    v: float,
    from_bounds: Bounds,
    to_bounds: Bounds,
) -> Tuple[float, float]:
    f_min_u, f_min_v, f_max_u, f_max_v = _normalize_bounds(from_bounds)
    t_min_u, t_min_v, t_max_u, t_max_v = _normalize_bounds(to_bounds)

    f_w = f_max_u - f_min_u
    f_h = f_max_v - f_min_v
    t_w = t_max_u - t_min_u
    t_h = t_max_v - t_min_v

    if f_w <= _GEOM_EPSILON or f_h <= _GEOM_EPSILON:
        return (u, v)

    t_u = (u - f_min_u) / f_w
    t_v = (v - f_min_v) / f_h
    return (t_min_u + t_u * t_w, t_min_v + t_v * t_h)


def _remap_triangles_between_bounds(
    triangles: Sequence[TriangleUV],
    from_bounds: Bounds,
    to_bounds: Bounds,
) -> list:
    remapped = []
    for tri in triangles:
        u0, v0, u1, v1, u2, v2 = tri
        ru0, rv0 = _remap_uv_between_bounds(u0, v0, from_bounds, to_bounds)
        ru1, rv1 = _remap_uv_between_bounds(u1, v1, from_bounds, to_bounds)
        ru2, rv2 = _remap_uv_between_bounds(u2, v2, from_bounds, to_bounds)
        remapped.append((ru0, rv0, ru1, rv1, ru2, rv2))
    return remapped


def _triangle_precompute(tri: TriangleUV) -> Optional[Tuple[float, ...]]:
    u0, v0, u1, v1, u2, v2 = tri
    a = v1 - v2
    b = u2 - u1
    c = v2 - v0
    d = u0 - u2
    denom = a * (u0 - u2) + b * (v0 - v2)
    if abs(denom) <= _GEOM_EPSILON:
        return None

    inv_denom = 1.0 / denom
    return (
        u0,
        v0,
        u1,
        v1,
        u2,
        v2,
        a,
        b,
        c,
        d,
        inv_denom,
        min(u0, u1, u2),
        min(v0, v1, v2),
        max(u0, u1, u2),
        max(v0, v1, v2),
    )


def _triangle_pre_bounds(pre: Tuple[float, ...]) -> Bounds:
    return (pre[11], pre[12], pre[13], pre[14])


def _triangle_pre_weights(u: float, v: float, pre: Tuple[float, ...]) -> Optional[Tuple[float, float, float]]:
    u0, v0, u1, v1, u2, v2, a, b, c, d, inv_denom, min_u, min_v, max_u, max_v = pre
    if u < min_u - _UV_TOLERANCE or u > max_u + _UV_TOLERANCE:
        return None
    if v < min_v - _UV_TOLERANCE or v > max_v + _UV_TOLERANCE:
        return None

    du = u - u2
    dv = v - v2
    w0 = (a * du + b * dv) * inv_denom
    w1 = (c * du + d * dv) * inv_denom
    w2 = 1.0 - w0 - w1

    if w0 < -_UV_TOLERANCE or w1 < -_UV_TOLERANCE or w2 < -_UV_TOLERANCE:
        return None
    if w0 > 1.0 + _UV_TOLERANCE or w1 > 1.0 + _UV_TOLERANCE or w2 > 1.0 + _UV_TOLERANCE:
        return None
    return (w0, w1, w2)


def _build_triangle_lookup_from_precomputed(
    precomputed: Sequence[Tuple[float, ...]],
    width: int,
    height: int,
    bin_size: int = _TRI_BIN_SIZE,
) -> Dict[str, object]:
    bins: Dict[Tuple[int, int], list] = {}
    cell_size = max(1, int(bin_size))

    for tri_index, pre in enumerate(precomputed):
        x0, y0, x1, y1 = _bounds_to_pixel_window(_triangle_pre_bounds(pre), width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        bx0 = x0 // cell_size
        by0 = y0 // cell_size
        bx1 = (x1 - 1) // cell_size
        by1 = (y1 - 1) // cell_size

        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                key = (bx, by)
                if key not in bins:
                    bins[key] = [tri_index]
                else:
                    bins[key].append(tri_index)

    return {"pre": list(precomputed), "bins": bins, "bin_size": cell_size}


def _build_triangle_lookup(
    triangles: Sequence[TriangleUV],
    width: int,
    height: int,
    bin_size: int = _TRI_BIN_SIZE,
) -> Dict[str, object]:
    precomputed = []
    for tri in triangles:
        pre = _triangle_precompute(tri)
        if pre is not None:
            precomputed.append(pre)
    return _build_triangle_lookup_from_precomputed(precomputed, width, height, bin_size)


def _triangle_lookup_hit(
    lookup: Dict[str, object],
    u: float,
    v: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, Tuple[float, float, float]]]:
    precomputed = lookup["pre"]
    if not precomputed:
        return None

    bins = lookup["bins"]
    bin_size = lookup["bin_size"]

    x = int(_clamp(u * width, 0.0, float(max(0, width - 1))))
    y = int(_clamp(v * height, 0.0, float(max(0, height - 1))))
    tri_candidates = bins.get((x // bin_size, y // bin_size))
    if not tri_candidates:
        return None

    for tri_index in tri_candidates:
        weights = _triangle_pre_weights(u, v, precomputed[tri_index])
        if weights is not None:
            return (tri_index, weights)
    return None


def _precomputed_union_bounds(precomputed: Sequence[Tuple[float, ...]]) -> Optional[Bounds]:
    if not precomputed:
        return None

    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    for pre in precomputed:
        min_u = min(min_u, pre[11])
        min_v = min(min_v, pre[12])
        max_u = max(max_u, pre[13])
        max_v = max(max_v, pre[14])
    return (min_u, min_v, max_u, max_v)


def _selected_uv_islands(context: bpy.types.Context) -> Optional[list]:
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        return None

    selected_face_loop_slots: Dict[int, list] = {}

    for face in bm.faces:
        if face.hide:
            continue

        selected_slots = []
        for slot, loop in enumerate(face.loops):
            luv = loop[uv_layer]
            if _loop_selected(context, face, loop, luv):
                selected_slots.append(slot)

        if selected_slots:
            selected_face_loop_slots[face.index] = selected_slots

    if not selected_face_loop_slots:
        return None

    selected_face_indices = set(selected_face_loop_slots.keys())
    adjacency: Dict[int, set] = {face_index: set() for face_index in selected_face_indices}

    for face_index in selected_face_indices:
        face = bm.faces[face_index]
        for edge in face.edges:
            for other_face in edge.link_faces:
                other_index = other_face.index
                if other_index == face_index or other_index not in selected_face_indices:
                    continue

                if _faces_uv_connected(face, other_face, edge, uv_layer):
                    adjacency[face_index].add(other_index)

    islands = []
    visited = set()

    for start_index in selected_face_indices:
        if start_index in visited:
            continue

        stack = [start_index]
        component_faces = []
        visited.add(start_index)

        while stack:
            current = stack.pop()
            component_faces.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        loop_refs = []
        for face_index in component_faces:
            for slot in selected_face_loop_slots[face_index]:
                loop_refs.append((face_index, slot))

        face_indices = sorted(component_faces)
        sample_triangles = _face_uv_triangles_from_face_indices(bm, uv_layer, face_indices)
        if sample_triangles is None or len(sample_triangles) == 0:
            continue

        bounds = _triangles_bounds(sample_triangles)
        if bounds is None:
            continue

        islands.append(
            {
                "loop_refs": loop_refs,
                "face_indices": face_indices,
                "sample_triangles": sample_triangles,
                "sample_bounds": bounds,
                "source_bounds": bounds,
            }
        )

    return islands if islands else None


def _find_active_image(context: bpy.types.Context) -> Optional[bpy.types.Image]:
    space = context.space_data
    if space and space.type == "IMAGE_EDITOR" and space.image:
        return space.image

    obj = context.active_object
    if not obj:
        return None

    mat = obj.active_material
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None

    active_node = mat.node_tree.nodes.active
    if active_node and active_node.type == "TEX_IMAGE" and active_node.image:
        return active_node.image

    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image and node.select:
            return node.image

    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            return node.image

    return None


def _validate_image(image: bpy.types.Image) -> Optional[str]:
    if image.source == "TILED":
        return "UDIM tiled images are not supported by this addon."

    width = int(image.size[0])
    height = int(image.size[1])
    if width <= 0 or height <= 0:
        return "Image has invalid dimensions."

    channels = int(image.channels)
    if channels <= 0:
        return "Image has no color channels."

    return None


def _pixel_index(x: int, y: int, width: int, channels: int) -> int:
    return (y * width + x) * channels


def _sample_bilinear(
    pixels: Sequence[float],
    width: int,
    height: int,
    channels: int,
    u: float,
    v: float,
) -> Tuple[float, ...]:
    x = _clamp(u * (width - 1), 0.0, float(width - 1))
    y = _clamp(v * (height - 1), 0.0, float(height - 1))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    tx = x - x0
    ty = y - y0

    i00 = _pixel_index(x0, y0, width, channels)
    i10 = _pixel_index(x1, y0, width, channels)
    i01 = _pixel_index(x0, y1, width, channels)
    i11 = _pixel_index(x1, y1, width, channels)

    out = []
    for c in range(channels):
        c00 = float(pixels[i00 + c])
        c10 = float(pixels[i10 + c])
        c01 = float(pixels[i01 + c])
        c11 = float(pixels[i11 + c])
        top = c00 + (c10 - c00) * tx
        bottom = c01 + (c11 - c01) * tx
        out.append(top + (bottom - top) * ty)
    return tuple(out)


def _bounds_to_pixel_window(bounds: Bounds, width: int, height: int) -> Tuple[int, int, int, int]:
    min_u, min_v, max_u, max_v = _normalize_bounds(bounds)
    x0 = max(0, int(math.floor(min_u * width)))
    y0 = max(0, int(math.floor(min_v * height)))
    x1 = min(width, int(math.ceil(max_u * width)))
    y1 = min(height, int(math.ceil(max_v * height)))
    return x0, y0, x1, y1


def _clear_region(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    bounds: Bounds,
) -> None:
    x0, y0, x1, y1 = _bounds_to_pixel_window(bounds, width, height)
    if x1 <= x0 or y1 <= y0:
        return

    for y in range(y0, y1):
        for x in range(x0, x1):
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = 0.0


def _clear_region_masked(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    bounds: Bounds,
    triangles: Sequence[TriangleUV],
) -> None:
    x0, y0, x1, y1 = _bounds_to_pixel_window(bounds, width, height)
    if x1 <= x0 or y1 <= y0:
        return

    lookup = _build_triangle_lookup(triangles, width, height)
    for y in range(y0, y1):
        v = (y + 0.5) / height
        for x in range(x0, x1):
            u = (x + 0.5) / width
            if _triangle_lookup_hit(lookup, u, v, width, height) is None:
                continue
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = 0.0


def _expand_bounds_by_pixels(bounds: Bounds, width: int, height: int, pad_px: float) -> Bounds:
    min_u, min_v, max_u, max_v = _normalize_bounds(bounds)
    if abs(pad_px) <= _EPSILON:
        return (min_u, min_v, max_u, max_v)

    pad_u = pad_px / max(1, width)
    pad_v = pad_px / max(1, height)
    return (min_u - pad_u, min_v - pad_v, max_u + pad_u, max_v + pad_v)


def _dilate_mask_square(mask: bytearray, width: int, height: int, margin_px: int) -> bytearray:
    if margin_px <= 0:
        return mask

    horizontal = bytearray(width * height)
    radius = int(margin_px)

    for y in range(height):
        row = y * width
        count = 0
        right = min(width - 1, radius)
        for x in range(0, right + 1):
            count += mask[row + x]

        for x in range(width):
            if count > 0:
                horizontal[row + x] = 1

            left_x = x - radius
            right_x = x + radius + 1
            if left_x >= 0:
                count -= mask[row + left_x]
            if right_x < width:
                count += mask[row + right_x]

    out = bytearray(width * height)
    for x in range(width):
        count = 0
        bottom = min(height - 1, radius)
        for y in range(0, bottom + 1):
            count += horizontal[y * width + x]

        for y in range(height):
            if count > 0:
                out[y * width + x] = 1

            top_y = y - radius
            bottom_y = y + radius + 1
            if top_y >= 0:
                count -= horizontal[top_y * width + x]
            if bottom_y < height:
                count += horizontal[bottom_y * width + x]

    return out


def _bleed_islands_non_interfering(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    island_masks: Sequence[bytearray],
    margin_px: int,
    soft_tie_blend: bool = True,
) -> None:
    if margin_px <= 0 or not island_masks:
        return

    pixel_count = width * height
    owner = [-1] * pixel_count
    dist = [margin_px + 1] * pixel_count
    source_idx = [-1] * pixel_count
    queue = deque()

    for island_id, mask in enumerate(island_masks):
        if mask is None or len(mask) != pixel_count:
            continue
        for idx, value in enumerate(mask):
            if not value:
                continue
            if owner[idx] == -1:
                owner[idx] = island_id
                dist[idx] = 0
                source_idx[idx] = idx
                queue.append(idx)

    if not queue:
        return

    while queue:
        idx = queue.popleft()
        current_dist = dist[idx]
        if current_dist >= margin_px:
            continue

        x = idx % width
        y = idx // width
        next_dist = current_dist + 1

        for ny in range(max(0, y - 1), min(height - 1, y + 1) + 1):
            row_start = ny * width
            for nx in range(max(0, x - 1), min(width - 1, x + 1) + 1):
                if nx == x and ny == y:
                    continue
                nidx = row_start + nx

                if owner[nidx] == -1:
                    owner[nidx] = owner[idx]
                    dist[nidx] = next_dist
                    source_idx[nidx] = source_idx[idx]
                    queue.append(nidx)
                elif next_dist < dist[nidx]:
                    owner[nidx] = owner[idx]
                    dist[nidx] = next_dist
                    source_idx[nidx] = source_idx[idx]
                    queue.append(nidx)

    base_pixels = list(out_pixels)
    for idx in range(pixel_count):
        if owner[idx] == -1 or dist[idx] <= 0 or dist[idx] > margin_px:
            continue

        if not soft_tie_blend:
            src = source_idx[idx]
            if src < 0:
                continue
            dst_pixel = idx * channels
            src_pixel = src * channels
            for c in range(channels):
                out_pixels[dst_pixel + c] = base_pixels[src_pixel + c]
            continue

        tied_sources = {source_idx[idx]}
        x = idx % width
        y = idx // width

        # Soft blend at border ties: if neighboring pixels at the same distance
        # are owned by different islands, include their source colors too.
        for ny in range(max(0, y - 1), min(height - 1, y + 1) + 1):
            row_start = ny * width
            for nx in range(max(0, x - 1), min(width - 1, x + 1) + 1):
                if nx == x and ny == y:
                    continue
                nidx = row_start + nx
                if owner[nidx] == -1:
                    continue
                if dist[nidx] != dist[idx]:
                    continue
                if owner[nidx] != owner[idx] and source_idx[nidx] >= 0:
                    tied_sources.add(source_idx[nidx])

        tied_sources = [s for s in tied_sources if s >= 0]
        if not tied_sources:
            continue

        dst_pixel = idx * channels
        inv_count = 1.0 / len(tied_sources)
        for c in range(channels):
            accum = 0.0
            for src in tied_sources:
                accum += base_pixels[src * channels + c]
            out_pixels[dst_pixel + c] = accum * inv_count


def _clear_unused_space(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    used_bounds: Sequence[Bounds],
    used_triangles: Sequence[Sequence[TriangleUV]],
    margin_px: int,
    sheet_edge_margin_px: int = 0,
) -> None:
    base_keep = bytearray(width * height)

    for island_index, bounds in enumerate(used_bounds):
        triangles: Sequence[TriangleUV] = []
        if island_index < len(used_triangles):
            triangles = used_triangles[island_index]

        x0, y0, x1, y1 = _bounds_to_pixel_window(bounds, width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        lookup = _build_triangle_lookup(triangles, width, height) if triangles else None
        for y in range(y0, y1):
            v = (y + 0.5) / height
            row_start = y * width
            for x in range(x0, x1):
                u = (x + 0.5) / width
                if lookup is not None and _triangle_lookup_hit(lookup, u, v, width, height) is None:
                    continue
                base_keep[row_start + x] = 1

    keep = _dilate_mask_square(base_keep, width, height, margin_px)
    border = max(0, int(sheet_edge_margin_px))
    if border > 0:
        border = min(border, width // 2, height // 2)
        if border > 0:
            for y in range(height):
                row_start = y * width
                for x in range(width):
                    if x < border or x >= (width - border) or y < border or y >= (height - border):
                        keep[row_start + x] = 1

    for y in range(height):
        row_start = y * width
        for x in range(width):
            if keep[row_start + x]:
                continue
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = 0.0


def _selected_keep_regions(context: bpy.types.Context) -> Optional[Tuple[list, list]]:
    islands = _selected_uv_islands(context)
    if not islands:
        return None

    used_bounds = []
    used_triangles = []
    for island in islands:
        triangles = island.get("sample_triangles")
        if not triangles:
            continue
        bounds = _triangles_bounds(triangles)
        if bounds is None:
            continue
        used_bounds.append(bounds)
        used_triangles.append(triangles)

    if not used_bounds:
        return None
    return (used_bounds, used_triangles)


def _transform_region_inplace(
    out_pixels: list,
    source_pixels: Sequence[float],
    width: int,
    height: int,
    channels: int,
    src_tex_bounds: Bounds,
    dst_tex_bounds: Bounds,
    bounds_pad_px: float,
    edge_samples: int,
    src_island_triangles: Sequence[TriangleUV],
    dst_island_triangles: Sequence[TriangleUV],
    write_mask: Optional[bytearray] = None,
    debug_out: Optional[Dict[str, object]] = None,
) -> bool:
    if debug_out is not None:
        debug_out.clear()
        debug_out["stage"] = "start"
        debug_out["src_triangles_input"] = len(src_island_triangles) if src_island_triangles else 0
        debug_out["dst_triangles_input"] = len(dst_island_triangles) if dst_island_triangles else 0
        debug_out["bounds_pad_px"] = float(bounds_pad_px)
        debug_out["edge_samples"] = int(edge_samples)

    if not src_island_triangles or not dst_island_triangles:
        if debug_out is not None:
            debug_out["stage"] = "validate_input"
            debug_out["reason"] = "empty_triangle_input"
        return False

    src_base_bounds = _normalize_bounds(src_tex_bounds)
    dst_base_bounds = _normalize_bounds(dst_tex_bounds)
    if debug_out is not None:
        debug_out["src_bounds"] = _format_bounds(src_base_bounds)
        debug_out["dst_bounds"] = _format_bounds(dst_base_bounds)
    if (src_base_bounds[2] - src_base_bounds[0]) <= _GEOM_EPSILON or (src_base_bounds[3] - src_base_bounds[1]) <= _GEOM_EPSILON:
        if debug_out is not None:
            debug_out["stage"] = "validate_bounds"
            debug_out["reason"] = "src_bounds_zero_area"
        return False
    if (dst_base_bounds[2] - dst_base_bounds[0]) <= _GEOM_EPSILON or (dst_base_bounds[3] - dst_base_bounds[1]) <= _GEOM_EPSILON:
        if debug_out is not None:
            debug_out["stage"] = "validate_bounds"
            debug_out["reason"] = "dst_bounds_zero_area"
        return False

    src_tris = list(src_island_triangles)
    dst_tris = list(dst_island_triangles)

    if abs(bounds_pad_px) > _EPSILON:
        src_padded_bounds = _expand_bounds_by_pixels(src_base_bounds, width, height, bounds_pad_px)
        dst_padded_bounds = _expand_bounds_by_pixels(dst_base_bounds, width, height, bounds_pad_px)
        src_tris = _remap_triangles_between_bounds(src_tris, src_base_bounds, src_padded_bounds)
        dst_tris = _remap_triangles_between_bounds(dst_tris, dst_base_bounds, dst_padded_bounds)

    tri_count = min(len(src_tris), len(dst_tris))
    if debug_out is not None:
        debug_out["tri_count_src"] = len(src_tris)
        debug_out["tri_count_dst"] = len(dst_tris)
        debug_out["tri_count_used"] = tri_count
    if tri_count == 0:
        if debug_out is not None:
            debug_out["stage"] = "triangulation"
            debug_out["reason"] = "no_triangles_after_setup"
        return False

    src_pre = []
    dst_pre = []
    for tri_index in range(tri_count):
        src_data = _triangle_precompute(src_tris[tri_index])
        dst_data = _triangle_precompute(dst_tris[tri_index])
        if src_data is None or dst_data is None:
            continue
        src_pre.append(src_data)
        dst_pre.append(dst_data)

    if not src_pre or len(src_pre) != len(dst_pre):
        if debug_out is not None:
            debug_out["stage"] = "triangulation"
            debug_out["reason"] = "no_valid_triangle_pairs_after_precompute"
            debug_out["src_pre"] = len(src_pre)
            debug_out["dst_pre"] = len(dst_pre)
        return False

    dst_lookup = _build_triangle_lookup_from_precomputed(dst_pre, width, height)
    dst_union_bounds = _precomputed_union_bounds(dst_pre)
    if dst_union_bounds is None:
        if debug_out is not None:
            debug_out["stage"] = "lookup"
            debug_out["reason"] = "no_destination_union_bounds"
        return False
    if debug_out is not None:
        debug_out["dst_union_bounds"] = _format_bounds(dst_union_bounds)
        debug_out["dst_bins"] = len(dst_lookup["bins"])
        debug_out["dst_pre"] = len(dst_lookup["pre"])

    x0, y0, x1, y1 = _bounds_to_pixel_window(dst_union_bounds, width, height)
    if debug_out is not None:
        debug_out["pixel_window"] = f"{x0},{y0} -> {x1},{y1}"

    if x1 <= x0 or y1 <= y0:
        if debug_out is not None:
            debug_out["stage"] = "window"
            debug_out["reason"] = "empty_destination_pixel_window"
        return True

    samples = max(1, int(edge_samples))
    inv_samples = 1.0 / samples
    attempts = 0
    hits = 0
    written_pixels = 0

    for y in range(y0, y1):
        for x in range(x0, x1):
            accum = [0.0] * channels
            valid = 0

            for sy in range(samples):
                dst_v = (y + (sy + 0.5) * inv_samples) / height

                for sx in range(samples):
                    dst_u = (x + (sx + 0.5) * inv_samples) / width
                    attempts += 1
                    hit = _triangle_lookup_hit(dst_lookup, dst_u, dst_v, width, height)
                    if hit is None:
                        continue

                    hits += 1
                    tri_index, weights = hit
                    w0, w1, w2 = weights
                    src_data = src_pre[tri_index]
                    src_u = w0 * src_data[0] + w1 * src_data[2] + w2 * src_data[4]
                    src_v = w0 * src_data[1] + w1 * src_data[3] + w2 * src_data[5]

                    sample = _sample_bilinear(source_pixels, width, height, channels, src_u, src_v)
                    for c in range(channels):
                        accum[c] += sample[c]
                    valid += 1

            if valid == 0:
                continue

            factor = 1.0 / valid
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = accum[c] * factor
            if write_mask is not None:
                write_mask[y * width + x] = 1
            written_pixels += 1

    if debug_out is not None:
        debug_out["stage"] = "done"
        debug_out["sample_attempts"] = attempts
        debug_out["sample_hits"] = hits
        debug_out["written_pixels"] = written_pixels
        if written_pixels == 0:
            debug_out["reason"] = "no_destination_pixels_written"
    return True


class UVTEXLINK_State(PropertyGroup):
    has_sample: BoolProperty(default=False)
    last_debug_report: StringProperty(default="")

    image_name: StringProperty(default="")
    image_width: IntProperty(default=0)
    image_height: IntProperty(default=0)
    image_channels: IntProperty(default=0)

    island_count: IntProperty(default=0, min=0)
    clear_old_region: BoolProperty(
        name="Clear Old Region",
        description="Clear each source texture region before writing rescaled result",
        default=False,
    )
    only_changed_islands: BoolProperty(
        name="Only Changed Islands",
        description="During confirm, process only islands whose UVs changed since sample",
        default=True,
    )
    uv_change_threshold: FloatProperty(
        name="Change Threshold",
        description="UV delta threshold used to detect changed islands",
        default=1e-8,
        min=0.0,
        soft_max=0.001,
        precision=8,
    )
    unused_margin_px: IntProperty(
        name="Unused Margin (px)",
        description="Pixel margin around used UV bounds when clearing unused texture space",
        default=4,
        min=0,
        soft_max=64,
    )
    uv_keep_margin_px: IntProperty(
        name="UV Keep Margin (px)",
        description="Additional preserved margin outside UV islands when clearing unused texture space",
        default=0,
        min=0,
        soft_max=64,
    )
    sheet_edge_margin_px: IntProperty(
        name="Sheet Edge Margin (px)",
        description="Protect this many pixels along texture-sheet edges from unused-space clearing",
        default=0,
        min=0,
        soft_max=256,
    )
    bounds_precision_px: FloatProperty(
        name="Bounds Pad (px)",
        description="Expand or shrink source/destination bounds in pixel units for cleaner edge sampling",
        default=0.25,
        min=-8.0,
        max=8.0,
        soft_min=-2.0,
        soft_max=2.0,
        precision=3,
    )
    edge_samples: IntProperty(
        name="Edge Samples",
        description="Sub-samples per axis for each destination texel (higher = cleaner edges, slower)",
        default=2,
        min=1,
        max=6,
    )
    uv_edge_bleed_px: IntProperty(
        name="UV Edge Bleed (px)",
        description="Bleed confirmed colors outward from UV island edges by this many pixels",
        default=0,
        min=0,
        soft_max=64,
    )
    soft_tie_blend: BoolProperty(
        name="Soft Blend Border Ties",
        description="When bleed fronts from different islands meet at the same distance, blend colors instead of hard ownership",
        default=True,
    )

    sampled_uv_min_u: FloatProperty(default=0.0)
    sampled_uv_min_v: FloatProperty(default=0.0)
    sampled_uv_max_u: FloatProperty(default=1.0)
    sampled_uv_max_v: FloatProperty(default=1.0)

    # Editable only for single-island samples.
    src_tex_min_u: FloatProperty(default=0.0)
    src_tex_min_v: FloatProperty(default=0.0)
    src_tex_max_u: FloatProperty(default=1.0)
    src_tex_max_v: FloatProperty(default=1.0)


def _set_uv_bounds_to_state(state: UVTEXLINK_State, uv_bounds: Bounds) -> None:
    uv_min_u, uv_min_v, uv_max_u, uv_max_v = uv_bounds
    state.sampled_uv_min_u = uv_min_u
    state.sampled_uv_min_v = uv_min_v
    state.sampled_uv_max_u = uv_max_u
    state.sampled_uv_max_v = uv_max_v


def _set_src_tex_bounds_to_state(state: UVTEXLINK_State, tex_bounds: Bounds) -> None:
    tex_min_u, tex_min_v, tex_max_u, tex_max_v = tex_bounds
    state.src_tex_min_u = tex_min_u
    state.src_tex_min_v = tex_min_v
    state.src_tex_max_u = tex_max_u
    state.src_tex_max_v = tex_max_v


def _get_state_uv_bounds(state: UVTEXLINK_State) -> Bounds:
    return (
        state.sampled_uv_min_u,
        state.sampled_uv_min_v,
        state.sampled_uv_max_u,
        state.sampled_uv_max_v,
    )


def _get_state_tex_bounds(state: UVTEXLINK_State) -> Bounds:
    return (
        state.src_tex_min_u,
        state.src_tex_min_v,
        state.src_tex_max_u,
        state.src_tex_max_v,
    )


class UVTEXLINK_OT_sample_state(Operator):
    bl_idname = "uvtexlink.sample_state"
    bl_label = "Sample UV + Texture"
    bl_description = "Sample selected UV islands and snapshot the active image"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def execute(self, context: bpy.types.Context):
        islands = _selected_uv_islands(context)
        if islands is None:
            self.report({"ERROR"}, "No UV selection found in Edit Mode.")
            return {"CANCELLED"}

        image = _find_active_image(context)
        if image is None:
            self.report({"ERROR"}, "No active image found in Image Editor or active material texture node.")
            return {"CANCELLED"}

        validation_error = _validate_image(image)
        if validation_error:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)
        snapshot = list(image.pixels[:])

        scene_key = context.scene.as_pointer()
        _PIXEL_BUFFERS[scene_key] = {
            "image_name": image.name,
            "width": width,
            "height": height,
            "channels": channels,
            "pixels": snapshot,
            "islands": islands,
        }

        state = context.scene.uvtexlink_state
        state.has_sample = True
        state.image_name = image.name
        state.image_width = width
        state.image_height = height
        state.image_channels = channels
        state.island_count = len(islands)
        state.last_debug_report = ""

        island_bounds = [island["sample_bounds"] for island in islands]
        overall_bounds = _union_bounds(island_bounds)
        if overall_bounds is not None:
            _set_uv_bounds_to_state(state, overall_bounds)

        if len(islands) == 1:
            _set_src_tex_bounds_to_state(state, islands[0]["source_bounds"])

        self.report(
            {"INFO"},
            f"Sampled {len(islands)} UV island(s) and buffered image '{image.name}'.",
        )
        return {"FINISHED"}


class UVTEXLINK_OT_confirm_rescale(Operator):
    bl_idname = "uvtexlink.confirm_rescale"
    bl_label = "Confirm Rescale"
    bl_description = "Rescale buffered texture regions to each island's current UV bounds"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def execute(self, context: bpy.types.Context):
        state = context.scene.uvtexlink_state
        state.last_debug_report = ""
        if not state.has_sample:
            self.report({"ERROR"}, "No sample in buffer. Click 'Sample UV + Texture' first.")
            return {"CANCELLED"}

        scene_key = context.scene.as_pointer()
        buffer = _PIXEL_BUFFERS.get(scene_key)
        if not buffer:
            self.report({"ERROR"}, "Pixel buffer is empty. Re-sample before confirming.")
            state.has_sample = False
            return {"CANCELLED"}

        image = bpy.data.images.get(state.image_name)
        if image is None:
            self.report({"ERROR"}, f"Image '{state.image_name}' no longer exists.")
            return {"CANCELLED"}

        validation_error = _validate_image(image)
        if validation_error:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)

        if (
            width != int(buffer["width"])
            or height != int(buffer["height"])
            or channels != int(buffer["channels"])
        ):
            self.report({"ERROR"}, "Image dimensions/channels changed since sample. Re-sample first.")
            return {"CANCELLED"}

        islands = buffer.get("islands")
        if not islands:
            self.report({"ERROR"}, "No sampled island data found. Re-sample first.")
            return {"CANCELLED"}

        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh in Edit Mode.")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            self.report({"ERROR"}, "Active mesh has no UV layer.")
            return {"CANCELLED"}

        out_pixels = list(image.pixels[:])
        island_transfer_margin = int(state.uv_edge_bleed_px)
        island_masks_for_bleed = []
        only_changed = bool(state.only_changed_islands)
        uv_change_threshold = float(state.uv_change_threshold)
        processed_islands = 0
        skipped_unchanged_islands = 0

        for island_index, island in enumerate(islands):
            loop_refs = island.get("loop_refs")
            if not loop_refs:
                self.report({"ERROR"}, f"Sampled island {island_index + 1} has no loop refs.")
                return {"CANCELLED"}
            face_indices = island.get("face_indices")
            sample_triangles = island.get("sample_triangles")
            if not face_indices or not sample_triangles:
                self.report(
                    {"ERROR"},
                    f"Sampled island {island_index + 1} is missing per-pixel island data. Re-sample first.",
                )
                return {"CANCELLED"}

            dst_triangles = _face_uv_triangles_from_face_indices(bm, uv_layer, face_indices)
            if dst_triangles is None or len(dst_triangles) == 0:
                self.report(
                    {"ERROR"},
                    "Mesh topology changed since sample (face refs invalid). Re-sample first.",
                )
                return {"CANCELLED"}
            dst_uv_bounds = _triangles_bounds(dst_triangles)
            if dst_uv_bounds is None:
                self.report(
                    {"ERROR"},
                    "Mesh topology changed since sample (destination island bounds invalid). Re-sample first.",
                )
                return {"CANCELLED"}
            if len(islands) == 1:
                src_tex_bounds = _get_state_tex_bounds(state)
            else:
                src_tex_bounds = island.get("source_bounds")
                if src_tex_bounds is None:
                    self.report({"ERROR"}, f"Island {island_index + 1} is missing source bounds.")
                    return {"CANCELLED"}

            sampled_source_bounds = island.get("source_bounds")
            src_bounds_now = _normalize_bounds(src_tex_bounds)
            if (
                sampled_source_bounds is not None
                and ((src_bounds_now[2] - src_bounds_now[0]) <= _GEOM_EPSILON or (src_bounds_now[3] - src_bounds_now[1]) <= _GEOM_EPSILON)
            ):
                src_tex_bounds = sampled_source_bounds
                if len(islands) == 1:
                    _set_src_tex_bounds_to_state(state, src_tex_bounds)
                self.report(
                    {"WARNING"},
                    f"Island {island_index + 1}: source bounds were zero-area; reverted to sampled source bounds.",
                )

            use_source_mask = True
            src_triangles_for_transform = list(sample_triangles)
            source_changed = False

            if len(islands) == 1 and sampled_source_bounds is not None:
                if not _bounds_almost_equal(src_tex_bounds, sampled_source_bounds):
                    src_triangles_for_transform = _remap_triangles_between_bounds(
                        sample_triangles,
                        sampled_source_bounds,
                        src_tex_bounds,
                    )
                    use_source_mask = False
                    source_changed = True

            uv_delta = _triangles_max_delta(sample_triangles, dst_triangles)
            effective_threshold = max(uv_change_threshold, _GEOM_EPSILON)
            uv_changed = uv_delta >= effective_threshold
            if only_changed and not uv_changed and not source_changed:
                skipped_unchanged_islands += 1
                continue

            if state.clear_old_region:
                if use_source_mask:
                    _clear_region_masked(
                        out_pixels=out_pixels,
                        width=width,
                        height=height,
                        channels=channels,
                        bounds=src_tex_bounds,
                        triangles=sample_triangles,
                    )
                else:
                    _clear_region(out_pixels, width, height, channels, src_tex_bounds)

            island_mask = bytearray(width * height) if island_transfer_margin > 0 else None
            debug_out: Dict[str, object] = {}
            success = _transform_region_inplace(
                out_pixels=out_pixels,
                source_pixels=buffer["pixels"],
                width=width,
                height=height,
                channels=channels,
                src_tex_bounds=src_tex_bounds,
                dst_tex_bounds=dst_uv_bounds,
                bounds_pad_px=float(state.bounds_precision_px),
                edge_samples=int(state.edge_samples),
                src_island_triangles=src_triangles_for_transform,
                dst_island_triangles=dst_triangles,
                write_mask=island_mask,
                debug_out=debug_out,
            )
            if not success:
                debug_out["island_index"] = island_index + 1
                debug_out["src_bounds_runtime"] = _format_bounds(src_tex_bounds)
                debug_out["dst_bounds_runtime"] = _format_bounds(dst_uv_bounds)
                debug_text = _debug_compact(debug_out)
                state.last_debug_report = debug_text
                print(f"[UVTEXLINK DEBUG] {debug_text}")
                self.report(
                    {"ERROR"},
                    f"Island {island_index + 1} mapping failed. See 'Last Debug Report' in the addon panel.",
                )
                return {"CANCELLED"}
            processed_islands += 1

            if island_transfer_margin > 0 and island_mask is not None:
                island_masks_for_bleed.append(island_mask)

        if island_transfer_margin > 0 and island_masks_for_bleed:
            _bleed_islands_non_interfering(
                out_pixels=out_pixels,
                width=width,
                height=height,
                channels=channels,
                island_masks=island_masks_for_bleed,
                margin_px=island_transfer_margin,
                soft_tie_blend=bool(state.soft_tie_blend),
            )

        if processed_islands > 0:
            image.pixels = out_pixels
            image.update()

        state.has_sample = False
        state.island_count = 0
        state.last_debug_report = ""
        _PIXEL_BUFFERS.pop(scene_key, None)

        if processed_islands == 0 and only_changed:
            self.report({"INFO"}, f"No changed islands detected; skipped {skipped_unchanged_islands} island(s).")
        elif only_changed:
            self.report(
                {"INFO"},
                f"Rescaled {processed_islands} island(s); skipped {skipped_unchanged_islands} unchanged island(s).",
            )
        else:
            self.report(
                {"INFO"},
                f"Rescaled {processed_islands} sampled island(s) to their current UV bounds.",
            )
        return {"FINISHED"}


class UVTEXLINK_OT_clear_unused_texture_space(Operator):
    bl_idname = "uvtexlink.clear_unused_texture_space"
    bl_label = "Clear Unused Texture Space"
    bl_description = "Clear all texture pixels outside selected UV island regions using configured margins"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def execute(self, context: bpy.types.Context):
        state = context.scene.uvtexlink_state

        image = _find_active_image(context)
        if image is None and state.image_name:
            image = bpy.data.images.get(state.image_name)
        if image is None:
            self.report({"ERROR"}, "No active image found for unused-space cleanup.")
            return {"CANCELLED"}

        validation_error = _validate_image(image)
        if validation_error:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        keep_regions = _selected_keep_regions(context)
        if keep_regions is None:
            self.report({"ERROR"}, "Select one or more UV islands to keep before cleanup.")
            return {"CANCELLED"}

        used_bounds, used_triangles = keep_regions
        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)

        keep_margin = int(state.unused_margin_px) + int(state.uv_keep_margin_px)
        if state.bounds_precision_px > 0.0:
            keep_margin += int(math.ceil(state.bounds_precision_px))

        out_pixels = list(image.pixels[:])
        _clear_unused_space(
            out_pixels=out_pixels,
            width=width,
            height=height,
            channels=channels,
            used_bounds=used_bounds,
            used_triangles=used_triangles,
            margin_px=keep_margin,
            sheet_edge_margin_px=int(state.sheet_edge_margin_px),
        )

        image.pixels = out_pixels
        image.update()
        self.report({"INFO"}, f"Cleared unused texture space using {len(used_bounds)} selected island region(s).")
        return {"FINISHED"}


class UVTEXLINK_OT_clear_buffer(Operator):
    bl_idname = "uvtexlink.clear_buffer"
    bl_label = "Clear Buffer"
    bl_description = "Clear sampled UV/texture state and pixel buffer"

    def execute(self, context: bpy.types.Context):
        state = context.scene.uvtexlink_state
        state.has_sample = False
        state.image_name = ""
        state.image_width = 0
        state.image_height = 0
        state.image_channels = 0
        state.island_count = 0
        state.last_debug_report = ""

        scene_key = context.scene.as_pointer()
        _PIXEL_BUFFERS.pop(scene_key, None)

        self.report({"INFO"}, "UV/texture buffer cleared.")
        return {"FINISHED"}


def _draw_uvtexlink_ui(layout: bpy.types.UILayout, context: bpy.types.Context) -> None:
    state = context.scene.uvtexlink_state
    active_image = _find_active_image(context)

    col = layout.column(align=True)
    col.operator(UVTEXLINK_OT_sample_state.bl_idname, icon="IMPORT")
    col.operator(UVTEXLINK_OT_confirm_rescale.bl_idname, icon="CHECKMARK")
    col.operator(UVTEXLINK_OT_clear_unused_texture_space.bl_idname, icon="BRUSH_DATA")
    col.operator(UVTEXLINK_OT_clear_buffer.bl_idname, icon="TRASH")

    layout.separator()

    layout.label(text="1) Select UV island(s)")
    layout.label(text="2) Sample")
    layout.label(text="3) Move/scale UV")
    layout.label(text="4) Confirm")

    layout.prop(state, "clear_old_region", text="Clear Old Source Region")
    layout.prop(state, "only_changed_islands", text="Only Changed Islands")
    if state.only_changed_islands:
        layout.prop(state, "uv_change_threshold", text="Change Threshold")
    precision_box = layout.box()
    precision_box.label(text="Bounds Precision")
    precision_box.prop(state, "bounds_precision_px", text="Bounds Pad (px)")
    precision_box.prop(state, "edge_samples", text="Edge Samples")
    precision_box.prop(state, "uv_edge_bleed_px", text="UV Edge Bleed (px)")
    precision_box.prop(state, "soft_tie_blend", text="Soft Blend Border Ties")
    cleanup_box = layout.box()
    cleanup_box.label(text="Unused Space Cleanup")
    cleanup_box.prop(state, "unused_margin_px", text="Unused Margin (px)")
    cleanup_box.prop(state, "uv_keep_margin_px", text="UV Keep Margin (px)")
    cleanup_box.prop(state, "sheet_edge_margin_px", text="Sheet Edge Margin (px)")

    recommend_size = 0
    if active_image and int(active_image.size[0]) > 0 and int(active_image.size[1]) > 0:
        recommend_size = max(int(active_image.size[0]), int(active_image.size[1]))
    elif state.image_width > 0 and state.image_height > 0:
        recommend_size = max(int(state.image_width), int(state.image_height))

    if recommend_size > 0:
        rec_bleed = _recommended_bleed_px_for_resolution(recommend_size)
        rec_box = layout.box()
        rec_box.label(text=f"Recommended UV Edge Bleed ({recommend_size}px): {rec_bleed}px", icon="INFO")

    layout.separator()

    if state.last_debug_report:
        debug_box = layout.box()
        debug_box.label(text="Last Debug Report", icon="ERROR")
        for part in state.last_debug_report.split(" || "):
            debug_box.label(text=part[:140])

    if not state.has_sample:
        layout.label(text="Buffer: empty", icon="INFO")
        return

    layout.label(text=f"Image: {state.image_name}", icon="IMAGE_DATA")
    layout.label(text=f"Size: {state.image_width} x {state.image_height}")
    layout.label(text=f"Sampled islands: {state.island_count}")
    layout.label(text=f"Sampled UV: {_format_bounds(_get_state_uv_bounds(state))}")

    if state.island_count == 1:
        box = layout.box()
        box.label(text="Source Texture Bounds (Single Island)")
        row = box.row(align=True)
        row.prop(state, "src_tex_min_u", text="Min U")
        row.prop(state, "src_tex_min_v", text="Min V")
        row = box.row(align=True)
        row.prop(state, "src_tex_max_u", text="Max U")
        row.prop(state, "src_tex_max_v", text="Max V")
        box.label(text="Confirm maps source bounds")
        box.label(text="to this island's current UV bounds")
    else:
        box = layout.box()
        box.label(text="Multi-Island Mode")
        box.label(text="Each island keeps its own sampled")
        box.label(text="source bounds and remaps independently")


class UVTEXLINK_PT_panel_uv_editor(Panel):
    bl_label = "UV Texture Link"
    bl_idname = "UVTEXLINK_PT_panel_uv_editor"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UV Tex Link"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def draw(self, context: bpy.types.Context):
        _draw_uvtexlink_ui(self.layout, context)


def _draw_uv_menu(self, context: bpy.types.Context) -> None:
    if not _is_mesh_edit_mode(context):
        return
    layout = self.layout
    layout.separator()
    layout.operator(UVTEXLINK_OT_sample_state.bl_idname, icon="IMPORT")
    layout.operator(UVTEXLINK_OT_confirm_rescale.bl_idname, icon="CHECKMARK")
    layout.operator(UVTEXLINK_OT_clear_unused_texture_space.bl_idname, icon="BRUSH_DATA")
    layout.operator(UVTEXLINK_OT_clear_buffer.bl_idname, icon="TRASH")


classes = (
    UVTEXLINK_State,
    UVTEXLINK_OT_sample_state,
    UVTEXLINK_OT_confirm_rescale,
    UVTEXLINK_OT_clear_unused_texture_space,
    UVTEXLINK_OT_clear_buffer,
    UVTEXLINK_PT_panel_uv_editor,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.uvtexlink_state = PointerProperty(type=UVTEXLINK_State)
    if hasattr(bpy.types, "IMAGE_MT_uvs"):
        bpy.types.IMAGE_MT_uvs.append(_draw_uv_menu)


def unregister():
    if hasattr(bpy.types, "IMAGE_MT_uvs"):
        try:
            bpy.types.IMAGE_MT_uvs.remove(_draw_uv_menu)
        except ValueError:
            pass
    if hasattr(bpy.types.Scene, "uvtexlink_state"):
        del bpy.types.Scene.uvtexlink_state
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
k",
    "description": "Sample UV/texture bounds and rescale texture pixels to match moved UV islands",
    "category": "UV",
}

import math
from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import bmesh
import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup


# In-memory pixel snapshots keyed by scene pointer.
_PIXEL_BUFFERS: Dict[int, Dict[str, object]] = {}
_EPSILON = 1e-8
_GEOM_EPSILON = 1e-14
_UV_TOLERANCE = 1e-6
_TRI_BIN_SIZE = 32


Bounds = Tuple[float, float, float, float]
UVCoord = Tuple[float, float]
LoopRef = Tuple[int, int]
TriangleUV = Tuple[float, float, float, float, float, float]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_mesh_edit_mode(context: bpy.types.Context) -> bool:
    obj = context.active_object
    return bool(obj and obj.type == "MESH" and context.mode == "EDIT_MESH")


def _loop_selected(
    context: bpy.types.Context,
    face: bmesh.types.BMFace,
    loop: bmesh.types.BMLoop,
    luv: bmesh.types.BMLoopUV,
) -> bool:
    if not context.scene.tool_settings.use_uv_select_sync:
        return luv.select

    select_mode = context.scene.tool_settings.mesh_select_mode
    if select_mode[2]:
        return face.select
    if select_mode[1]:
        return loop.edge.select
    return loop.vert.select


def _normalize_bounds(bounds: Bounds) -> Bounds:
    min_u, min_v, max_u, max_v = bounds
    return (min(min_u, max_u), min(min_v, max_v), max(min_u, max_u), max(min_v, max_v))


def _format_bounds(bounds: Bounds) -> str:
    min_u, min_v, max_u, max_v = bounds
    return f"({min_u:.4f}, {min_v:.4f}) -> ({max_u:.4f}, {max_v:.4f})"


def _debug_compact(details: Dict[str, object]) -> str:
    ordered = []
    for key in sorted(details.keys()):
        ordered.append(f"{key}={details[key]}")
    return " || ".join(ordered)


def _bounds_almost_equal(a: Bounds, b: Bounds, eps: float = 1e-7) -> bool:
    a0, a1, a2, a3 = _normalize_bounds(a)
    b0, b1, b2, b3 = _normalize_bounds(b)
    return (
        abs(a0 - b0) <= eps
        and abs(a1 - b1) <= eps
        and abs(a2 - b2) <= eps
        and abs(a3 - b3) <= eps
    )


def _union_bounds(bounds_list: Sequence[Bounds]) -> Optional[Bounds]:
    if not bounds_list:
        return None

    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    for bounds in bounds_list:
        b_min_u, b_min_v, b_max_u, b_max_v = _normalize_bounds(bounds)
        min_u = min(min_u, b_min_u)
        min_v = min(min_v, b_min_v)
        max_u = max(max_u, b_max_u)
        max_v = max(max_v, b_max_v)

    return (min_u, min_v, max_u, max_v)


def _uv_equal(a: UVCoord, b: UVCoord, tolerance: float = _UV_TOLERANCE) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _edge_uv_pair(
    face: bmesh.types.BMFace,
    edge: bmesh.types.BMEdge,
    uv_layer: object,
) -> Optional[Tuple[UVCoord, UVCoord]]:
    for loop in face.loops:
        if loop.edge == edge:
            uv_a = loop[uv_layer].uv
            uv_b = loop.link_loop_next[uv_layer].uv
            return (float(uv_a.x), float(uv_a.y)), (float(uv_b.x), float(uv_b.y))
    return None


def _faces_uv_connected(
    face_a: bmesh.types.BMFace,
    face_b: bmesh.types.BMFace,
    edge: bmesh.types.BMEdge,
    uv_layer: object,
) -> bool:
    pair_a = _edge_uv_pair(face_a, edge, uv_layer)
    pair_b = _edge_uv_pair(face_b, edge, uv_layer)
    if pair_a is None or pair_b is None:
        return False

    a0, a1 = pair_a
    b0, b1 = pair_b
    return (_uv_equal(a0, b0) and _uv_equal(a1, b1)) or (_uv_equal(a0, b1) and _uv_equal(a1, b0))


def _bounds_from_loop_refs_bmesh(
    bm: bmesh.types.BMesh,
    uv_layer: object,
    loop_refs: Sequence[LoopRef],
) -> Optional[Bounds]:
    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")
    found = False

    for face_index, loop_slot in loop_refs:
        if face_index < 0 or face_index >= len(bm.faces):
            return None

        face = bm.faces[face_index]
        if loop_slot < 0 or loop_slot >= len(face.loops):
            return None

        uv = face.loops[loop_slot][uv_layer].uv
        found = True
        min_u = min(min_u, float(uv.x))
        min_v = min(min_v, float(uv.y))
        max_u = max(max_u, float(uv.x))
        max_v = max(max_v, float(uv.y))

    if not found:
        return None

    return (min_u, min_v, max_u, max_v)


def _face_uv_triangles_from_face_indices(
    bm: bmesh.types.BMesh,
    uv_layer: object,
    face_indices: Sequence[int],
) -> Optional[list]:
    triangles = []

    for face_index in face_indices:
        if face_index < 0 or face_index >= len(bm.faces):
            return None
        face = bm.faces[face_index]
        if len(face.loops) < 3:
            continue

        uv0 = face.loops[0][uv_layer].uv
        u0 = float(uv0.x)
        v0 = float(uv0.y)

        for i in range(1, len(face.loops) - 1):
            uv1 = face.loops[i][uv_layer].uv
            uv2 = face.loops[i + 1][uv_layer].uv
            triangles.append(
                (
                    u0,
                    v0,
                    float(uv1.x),
                    float(uv1.y),
                    float(uv2.x),
                    float(uv2.y),
                )
            )

    return triangles


def _point_in_triangle(u: float, v: float, tri: TriangleUV, eps: float = 1e-9) -> bool:
    weights = _barycentric_weights(u, v, tri)
    if weights is None:
        return False
    w0, w1, w2 = weights
    return (
        w0 >= -eps
        and w1 >= -eps
        and w2 >= -eps
        and w0 <= 1.0 + eps
        and w1 <= 1.0 + eps
        and w2 <= 1.0 + eps
    )


def _point_in_triangles(u: float, v: float, triangles: Sequence[TriangleUV]) -> bool:
    for tri in triangles:
        if _point_in_triangle(u, v, tri):
            return True
    return False


def _triangle_bounds(tri: TriangleUV) -> Bounds:
    u0, v0, u1, v1, u2, v2 = tri
    return (
        min(u0, u1, u2),
        min(v0, v1, v2),
        max(u0, u1, u2),
        max(v0, v1, v2),
    )


def _triangles_bounds(triangles: Sequence[TriangleUV]) -> Optional[Bounds]:
    if not triangles:
        return None

    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    for tri in triangles:
        b_min_u, b_min_v, b_max_u, b_max_v = _triangle_bounds(tri)
        min_u = min(min_u, b_min_u)
        min_v = min(min_v, b_min_v)
        max_u = max(max_u, b_max_u)
        max_v = max(max_v, b_max_v)

    return (min_u, min_v, max_u, max_v)


def _barycentric_weights(u: float, v: float, tri: TriangleUV) -> Optional[Tuple[float, float, float]]:
    u0, v0, u1, v1, u2, v2 = tri
    denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    if abs(denom) <= _GEOM_EPSILON:
        return None

    w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
    w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
    w2 = 1.0 - w0 - w1
    return (w0, w1, w2)


def _remap_uv_between_bounds(
    u: float,
    v: float,
    from_bounds: Bounds,
    to_bounds: Bounds,
) -> Tuple[float, float]:
    f_min_u, f_min_v, f_max_u, f_max_v = _normalize_bounds(from_bounds)
    t_min_u, t_min_v, t_max_u, t_max_v = _normalize_bounds(to_bounds)

    f_w = f_max_u - f_min_u
    f_h = f_max_v - f_min_v
    t_w = t_max_u - t_min_u
    t_h = t_max_v - t_min_v

    if f_w <= _GEOM_EPSILON or f_h <= _GEOM_EPSILON:
        return (u, v)

    t_u = (u - f_min_u) / f_w
    t_v = (v - f_min_v) / f_h
    return (t_min_u + t_u * t_w, t_min_v + t_v * t_h)


def _remap_triangles_between_bounds(
    triangles: Sequence[TriangleUV],
    from_bounds: Bounds,
    to_bounds: Bounds,
) -> list:
    remapped = []
    for tri in triangles:
        u0, v0, u1, v1, u2, v2 = tri
        ru0, rv0 = _remap_uv_between_bounds(u0, v0, from_bounds, to_bounds)
        ru1, rv1 = _remap_uv_between_bounds(u1, v1, from_bounds, to_bounds)
        ru2, rv2 = _remap_uv_between_bounds(u2, v2, from_bounds, to_bounds)
        remapped.append((ru0, rv0, ru1, rv1, ru2, rv2))
    return remapped


def _triangle_precompute(tri: TriangleUV) -> Optional[Tuple[float, ...]]:
    u0, v0, u1, v1, u2, v2 = tri
    a = v1 - v2
    b = u2 - u1
    c = v2 - v0
    d = u0 - u2
    denom = a * (u0 - u2) + b * (v0 - v2)
    if abs(denom) <= _GEOM_EPSILON:
        return None

    inv_denom = 1.0 / denom
    return (
        u0,
        v0,
        u1,
        v1,
        u2,
        v2,
        a,
        b,
        c,
        d,
        inv_denom,
        min(u0, u1, u2),
        min(v0, v1, v2),
        max(u0, u1, u2),
        max(v0, v1, v2),
    )


def _triangle_pre_bounds(pre: Tuple[float, ...]) -> Bounds:
    return (pre[11], pre[12], pre[13], pre[14])


def _triangle_pre_weights(u: float, v: float, pre: Tuple[float, ...]) -> Optional[Tuple[float, float, float]]:
    u0, v0, u1, v1, u2, v2, a, b, c, d, inv_denom, min_u, min_v, max_u, max_v = pre
    if u < min_u - _UV_TOLERANCE or u > max_u + _UV_TOLERANCE:
        return None
    if v < min_v - _UV_TOLERANCE or v > max_v + _UV_TOLERANCE:
        return None

    du = u - u2
    dv = v - v2
    w0 = (a * du + b * dv) * inv_denom
    w1 = (c * du + d * dv) * inv_denom
    w2 = 1.0 - w0 - w1

    if w0 < -_UV_TOLERANCE or w1 < -_UV_TOLERANCE or w2 < -_UV_TOLERANCE:
        return None
    if w0 > 1.0 + _UV_TOLERANCE or w1 > 1.0 + _UV_TOLERANCE or w2 > 1.0 + _UV_TOLERANCE:
        return None
    return (w0, w1, w2)


def _build_triangle_lookup_from_precomputed(
    precomputed: Sequence[Tuple[float, ...]],
    width: int,
    height: int,
    bin_size: int = _TRI_BIN_SIZE,
) -> Dict[str, object]:
    bins: Dict[Tuple[int, int], list] = {}
    cell_size = max(1, int(bin_size))

    for tri_index, pre in enumerate(precomputed):
        x0, y0, x1, y1 = _bounds_to_pixel_window(_triangle_pre_bounds(pre), width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        bx0 = x0 // cell_size
        by0 = y0 // cell_size
        bx1 = (x1 - 1) // cell_size
        by1 = (y1 - 1) // cell_size

        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                key = (bx, by)
                if key not in bins:
                    bins[key] = [tri_index]
                else:
                    bins[key].append(tri_index)

    return {"pre": list(precomputed), "bins": bins, "bin_size": cell_size}


def _build_triangle_lookup(
    triangles: Sequence[TriangleUV],
    width: int,
    height: int,
    bin_size: int = _TRI_BIN_SIZE,
) -> Dict[str, object]:
    precomputed = []
    for tri in triangles:
        pre = _triangle_precompute(tri)
        if pre is not None:
            precomputed.append(pre)
    return _build_triangle_lookup_from_precomputed(precomputed, width, height, bin_size)


def _triangle_lookup_hit(
    lookup: Dict[str, object],
    u: float,
    v: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, Tuple[float, float, float]]]:
    precomputed = lookup["pre"]
    if not precomputed:
        return None

    bins = lookup["bins"]
    bin_size = lookup["bin_size"]

    x = int(_clamp(u * width, 0.0, float(max(0, width - 1))))
    y = int(_clamp(v * height, 0.0, float(max(0, height - 1))))
    tri_candidates = bins.get((x // bin_size, y // bin_size))
    if not tri_candidates:
        return None

    for tri_index in tri_candidates:
        weights = _triangle_pre_weights(u, v, precomputed[tri_index])
        if weights is not None:
            return (tri_index, weights)
    return None


def _precomputed_union_bounds(precomputed: Sequence[Tuple[float, ...]]) -> Optional[Bounds]:
    if not precomputed:
        return None

    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    for pre in precomputed:
        min_u = min(min_u, pre[11])
        min_v = min(min_v, pre[12])
        max_u = max(max_u, pre[13])
        max_v = max(max_v, pre[14])
    return (min_u, min_v, max_u, max_v)


def _selected_uv_islands(context: bpy.types.Context) -> Optional[list]:
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        return None

    selected_face_loop_slots: Dict[int, list] = {}

    for face in bm.faces:
        if face.hide:
            continue

        selected_slots = []
        for slot, loop in enumerate(face.loops):
            luv = loop[uv_layer]
            if _loop_selected(context, face, loop, luv):
                selected_slots.append(slot)

        if selected_slots:
            selected_face_loop_slots[face.index] = selected_slots

    if not selected_face_loop_slots:
        return None

    selected_face_indices = set(selected_face_loop_slots.keys())
    adjacency: Dict[int, set] = {face_index: set() for face_index in selected_face_indices}

    for face_index in selected_face_indices:
        face = bm.faces[face_index]
        for edge in face.edges:
            for other_face in edge.link_faces:
                other_index = other_face.index
                if other_index == face_index or other_index not in selected_face_indices:
                    continue

                if _faces_uv_connected(face, other_face, edge, uv_layer):
                    adjacency[face_index].add(other_index)

    islands = []
    visited = set()

    for start_index in selected_face_indices:
        if start_index in visited:
            continue

        stack = [start_index]
        component_faces = []
        visited.add(start_index)

        while stack:
            current = stack.pop()
            component_faces.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        loop_refs = []
        for face_index in component_faces:
            for slot in selected_face_loop_slots[face_index]:
                loop_refs.append((face_index, slot))

        face_indices = sorted(component_faces)
        sample_triangles = _face_uv_triangles_from_face_indices(bm, uv_layer, face_indices)
        if sample_triangles is None or len(sample_triangles) == 0:
            continue

        bounds = _triangles_bounds(sample_triangles)
        if bounds is None:
            continue

        islands.append(
            {
                "loop_refs": loop_refs,
                "face_indices": face_indices,
                "sample_triangles": sample_triangles,
                "sample_bounds": bounds,
                "source_bounds": bounds,
            }
        )

    return islands if islands else None


def _find_active_image(context: bpy.types.Context) -> Optional[bpy.types.Image]:
    space = context.space_data
    if space and space.type == "IMAGE_EDITOR" and space.image:
        return space.image

    obj = context.active_object
    if not obj:
        return None

    mat = obj.active_material
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None

    active_node = mat.node_tree.nodes.active
    if active_node and active_node.type == "TEX_IMAGE" and active_node.image:
        return active_node.image

    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image and node.select:
            return node.image

    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            return node.image

    return None


def _validate_image(image: bpy.types.Image) -> Optional[str]:
    if image.source == "TILED":
        return "UDIM tiled images are not supported by this addon."

    width = int(image.size[0])
    height = int(image.size[1])
    if width <= 0 or height <= 0:
        return "Image has invalid dimensions."

    channels = int(image.channels)
    if channels <= 0:
        return "Image has no color channels."

    return None


def _pixel_index(x: int, y: int, width: int, channels: int) -> int:
    return (y * width + x) * channels


def _sample_bilinear(
    pixels: Sequence[float],
    width: int,
    height: int,
    channels: int,
    u: float,
    v: float,
) -> Tuple[float, ...]:
    x = _clamp(u * (width - 1), 0.0, float(width - 1))
    y = _clamp(v * (height - 1), 0.0, float(height - 1))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    tx = x - x0
    ty = y - y0

    i00 = _pixel_index(x0, y0, width, channels)
    i10 = _pixel_index(x1, y0, width, channels)
    i01 = _pixel_index(x0, y1, width, channels)
    i11 = _pixel_index(x1, y1, width, channels)

    out = []
    for c in range(channels):
        c00 = float(pixels[i00 + c])
        c10 = float(pixels[i10 + c])
        c01 = float(pixels[i01 + c])
        c11 = float(pixels[i11 + c])
        top = c00 + (c10 - c00) * tx
        bottom = c01 + (c11 - c01) * tx
        out.append(top + (bottom - top) * ty)
    return tuple(out)


def _bounds_to_pixel_window(bounds: Bounds, width: int, height: int) -> Tuple[int, int, int, int]:
    min_u, min_v, max_u, max_v = _normalize_bounds(bounds)
    x0 = max(0, int(math.floor(min_u * width)))
    y0 = max(0, int(math.floor(min_v * height)))
    x1 = min(width, int(math.ceil(max_u * width)))
    y1 = min(height, int(math.ceil(max_v * height)))
    return x0, y0, x1, y1


def _clear_region(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    bounds: Bounds,
) -> None:
    x0, y0, x1, y1 = _bounds_to_pixel_window(bounds, width, height)
    if x1 <= x0 or y1 <= y0:
        return

    for y in range(y0, y1):
        for x in range(x0, x1):
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = 0.0


def _clear_region_masked(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    bounds: Bounds,
    triangles: Sequence[TriangleUV],
) -> None:
    x0, y0, x1, y1 = _bounds_to_pixel_window(bounds, width, height)
    if x1 <= x0 or y1 <= y0:
        return

    lookup = _build_triangle_lookup(triangles, width, height)
    for y in range(y0, y1):
        v = (y + 0.5) / height
        for x in range(x0, x1):
            u = (x + 0.5) / width
            if _triangle_lookup_hit(lookup, u, v, width, height) is None:
                continue
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = 0.0


def _expand_bounds_by_pixels(bounds: Bounds, width: int, height: int, pad_px: float) -> Bounds:
    min_u, min_v, max_u, max_v = _normalize_bounds(bounds)
    if abs(pad_px) <= _EPSILON:
        return (min_u, min_v, max_u, max_v)

    pad_u = pad_px / max(1, width)
    pad_v = pad_px / max(1, height)
    return (min_u - pad_u, min_v - pad_v, max_u + pad_u, max_v + pad_v)


def _dilate_mask_square(mask: bytearray, width: int, height: int, margin_px: int) -> bytearray:
    if margin_px <= 0:
        return mask

    horizontal = bytearray(width * height)
    radius = int(margin_px)

    for y in range(height):
        row = y * width
        count = 0
        right = min(width - 1, radius)
        for x in range(0, right + 1):
            count += mask[row + x]

        for x in range(width):
            if count > 0:
                horizontal[row + x] = 1

            left_x = x - radius
            right_x = x + radius + 1
            if left_x >= 0:
                count -= mask[row + left_x]
            if right_x < width:
                count += mask[row + right_x]

    out = bytearray(width * height)
    for x in range(width):
        count = 0
        bottom = min(height - 1, radius)
        for y in range(0, bottom + 1):
            count += horizontal[y * width + x]

        for y in range(height):
            if count > 0:
                out[y * width + x] = 1

            top_y = y - radius
            bottom_y = y + radius + 1
            if top_y >= 0:
                count -= horizontal[top_y * width + x]
            if bottom_y < height:
                count += horizontal[bottom_y * width + x]

    return out


def _bleed_written_pixels(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    written_mask: bytearray,
    margin_px: int,
    bounds: Bounds,
) -> None:
    if margin_px <= 0:
        return

    expanded = _expand_bounds_by_pixels(bounds, width, height, float(margin_px))
    x0, y0, x1, y1 = _bounds_to_pixel_window(expanded, width, height)
    if x1 <= x0 or y1 <= y0:
        return

    local_width = x1 - x0
    local_height = y1 - y0
    local_visited = bytearray(local_width * local_height)
    queue = deque()

    for y in range(y0, y1):
        row_start = y * width
        local_row = (y - y0) * local_width
        for x in range(x0, x1):
            global_idx = row_start + x
            if not written_mask[global_idx]:
                continue
            local_idx = local_row + (x - x0)
            local_visited[local_idx] = 1
            queue.append((x, y, 0))

    if not queue:
        return

    while queue:
        x, y, dist = queue.popleft()
        if dist >= margin_px:
            continue

        src_pixel = _pixel_index(x, y, width, channels)
        for ny in range(max(y0, y - 1), min(y1 - 1, y + 1) + 1):
            local_row = (ny - y0) * local_width
            global_row = ny * width
            for nx in range(max(x0, x - 1), min(x1 - 1, x + 1) + 1):
                if nx == x and ny == y:
                    continue

                local_idx = local_row + (nx - x0)
                if local_visited[local_idx]:
                    continue

                dst_global_idx = global_row + nx
                dst_pixel = _pixel_index(nx, ny, width, channels)
                for c in range(channels):
                    out_pixels[dst_pixel + c] = out_pixels[src_pixel + c]

                local_visited[local_idx] = 1
                written_mask[dst_global_idx] = 1
                queue.append((nx, ny, dist + 1))


def _clear_unused_space(
    out_pixels: list,
    width: int,
    height: int,
    channels: int,
    used_bounds: Sequence[Bounds],
    used_triangles: Sequence[Sequence[TriangleUV]],
    margin_px: int,
    sheet_edge_margin_px: int = 0,
) -> None:
    base_keep = bytearray(width * height)

    for island_index, bounds in enumerate(used_bounds):
        triangles: Sequence[TriangleUV] = []
        if island_index < len(used_triangles):
            triangles = used_triangles[island_index]

        x0, y0, x1, y1 = _bounds_to_pixel_window(bounds, width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        lookup = _build_triangle_lookup(triangles, width, height) if triangles else None
        for y in range(y0, y1):
            v = (y + 0.5) / height
            row_start = y * width
            for x in range(x0, x1):
                u = (x + 0.5) / width
                if lookup is not None and _triangle_lookup_hit(lookup, u, v, width, height) is None:
                    continue
                base_keep[row_start + x] = 1

    keep = _dilate_mask_square(base_keep, width, height, margin_px)
    border = max(0, int(sheet_edge_margin_px))
    if border > 0:
        border = min(border, width // 2, height // 2)
        if border > 0:
            for y in range(height):
                row_start = y * width
                for x in range(width):
                    if x < border or x >= (width - border) or y < border or y >= (height - border):
                        keep[row_start + x] = 1

    for y in range(height):
        row_start = y * width
        for x in range(width):
            if keep[row_start + x]:
                continue
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = 0.0


def _selected_keep_regions(context: bpy.types.Context) -> Optional[Tuple[list, list]]:
    islands = _selected_uv_islands(context)
    if not islands:
        return None

    used_bounds = []
    used_triangles = []
    for island in islands:
        triangles = island.get("sample_triangles")
        if not triangles:
            continue
        bounds = _triangles_bounds(triangles)
        if bounds is None:
            continue
        used_bounds.append(bounds)
        used_triangles.append(triangles)

    if not used_bounds:
        return None
    return (used_bounds, used_triangles)


def _transform_region_inplace(
    out_pixels: list,
    source_pixels: Sequence[float],
    width: int,
    height: int,
    channels: int,
    src_tex_bounds: Bounds,
    dst_tex_bounds: Bounds,
    bounds_pad_px: float,
    edge_samples: int,
    src_island_triangles: Sequence[TriangleUV],
    dst_island_triangles: Sequence[TriangleUV],
    write_mask: Optional[bytearray] = None,
    debug_out: Optional[Dict[str, object]] = None,
) -> bool:
    if debug_out is not None:
        debug_out.clear()
        debug_out["stage"] = "start"
        debug_out["src_triangles_input"] = len(src_island_triangles) if src_island_triangles else 0
        debug_out["dst_triangles_input"] = len(dst_island_triangles) if dst_island_triangles else 0
        debug_out["bounds_pad_px"] = float(bounds_pad_px)
        debug_out["edge_samples"] = int(edge_samples)

    if not src_island_triangles or not dst_island_triangles:
        if debug_out is not None:
            debug_out["stage"] = "validate_input"
            debug_out["reason"] = "empty_triangle_input"
        return False

    src_base_bounds = _normalize_bounds(src_tex_bounds)
    dst_base_bounds = _normalize_bounds(dst_tex_bounds)
    if debug_out is not None:
        debug_out["src_bounds"] = _format_bounds(src_base_bounds)
        debug_out["dst_bounds"] = _format_bounds(dst_base_bounds)
    if (src_base_bounds[2] - src_base_bounds[0]) <= _GEOM_EPSILON or (src_base_bounds[3] - src_base_bounds[1]) <= _GEOM_EPSILON:
        if debug_out is not None:
            debug_out["stage"] = "validate_bounds"
            debug_out["reason"] = "src_bounds_zero_area"
        return False
    if (dst_base_bounds[2] - dst_base_bounds[0]) <= _GEOM_EPSILON or (dst_base_bounds[3] - dst_base_bounds[1]) <= _GEOM_EPSILON:
        if debug_out is not None:
            debug_out["stage"] = "validate_bounds"
            debug_out["reason"] = "dst_bounds_zero_area"
        return False

    src_tris = list(src_island_triangles)
    dst_tris = list(dst_island_triangles)

    if abs(bounds_pad_px) > _EPSILON:
        src_padded_bounds = _expand_bounds_by_pixels(src_base_bounds, width, height, bounds_pad_px)
        dst_padded_bounds = _expand_bounds_by_pixels(dst_base_bounds, width, height, bounds_pad_px)
        src_tris = _remap_triangles_between_bounds(src_tris, src_base_bounds, src_padded_bounds)
        dst_tris = _remap_triangles_between_bounds(dst_tris, dst_base_bounds, dst_padded_bounds)

    tri_count = min(len(src_tris), len(dst_tris))
    if debug_out is not None:
        debug_out["tri_count_src"] = len(src_tris)
        debug_out["tri_count_dst"] = len(dst_tris)
        debug_out["tri_count_used"] = tri_count
    if tri_count == 0:
        if debug_out is not None:
            debug_out["stage"] = "triangulation"
            debug_out["reason"] = "no_triangles_after_setup"
        return False

    src_pre = []
    dst_pre = []
    for tri_index in range(tri_count):
        src_data = _triangle_precompute(src_tris[tri_index])
        dst_data = _triangle_precompute(dst_tris[tri_index])
        if src_data is None or dst_data is None:
            continue
        src_pre.append(src_data)
        dst_pre.append(dst_data)

    if not src_pre or len(src_pre) != len(dst_pre):
        if debug_out is not None:
            debug_out["stage"] = "triangulation"
            debug_out["reason"] = "no_valid_triangle_pairs_after_precompute"
            debug_out["src_pre"] = len(src_pre)
            debug_out["dst_pre"] = len(dst_pre)
        return False

    dst_lookup = _build_triangle_lookup_from_precomputed(dst_pre, width, height)
    dst_union_bounds = _precomputed_union_bounds(dst_pre)
    if dst_union_bounds is None:
        if debug_out is not None:
            debug_out["stage"] = "lookup"
            debug_out["reason"] = "no_destination_union_bounds"
        return False
    if debug_out is not None:
        debug_out["dst_union_bounds"] = _format_bounds(dst_union_bounds)
        debug_out["dst_bins"] = len(dst_lookup["bins"])
        debug_out["dst_pre"] = len(dst_lookup["pre"])

    x0, y0, x1, y1 = _bounds_to_pixel_window(dst_union_bounds, width, height)
    if debug_out is not None:
        debug_out["pixel_window"] = f"{x0},{y0} -> {x1},{y1}"

    if x1 <= x0 or y1 <= y0:
        if debug_out is not None:
            debug_out["stage"] = "window"
            debug_out["reason"] = "empty_destination_pixel_window"
        return True

    samples = max(1, int(edge_samples))
    inv_samples = 1.0 / samples
    attempts = 0
    hits = 0
    written_pixels = 0

    for y in range(y0, y1):
        for x in range(x0, x1):
            accum = [0.0] * channels
            valid = 0

            for sy in range(samples):
                dst_v = (y + (sy + 0.5) * inv_samples) / height

                for sx in range(samples):
                    dst_u = (x + (sx + 0.5) * inv_samples) / width
                    attempts += 1
                    hit = _triangle_lookup_hit(dst_lookup, dst_u, dst_v, width, height)
                    if hit is None:
                        continue

                    hits += 1
                    tri_index, weights = hit
                    w0, w1, w2 = weights
                    src_data = src_pre[tri_index]
                    src_u = w0 * src_data[0] + w1 * src_data[2] + w2 * src_data[4]
                    src_v = w0 * src_data[1] + w1 * src_data[3] + w2 * src_data[5]

                    sample = _sample_bilinear(source_pixels, width, height, channels, src_u, src_v)
                    for c in range(channels):
                        accum[c] += sample[c]
                    valid += 1

            if valid == 0:
                continue

            factor = 1.0 / valid
            idx = _pixel_index(x, y, width, channels)
            for c in range(channels):
                out_pixels[idx + c] = accum[c] * factor
            if write_mask is not None:
                write_mask[y * width + x] = 1
            written_pixels += 1

    if debug_out is not None:
        debug_out["stage"] = "done"
        debug_out["sample_attempts"] = attempts
        debug_out["sample_hits"] = hits
        debug_out["written_pixels"] = written_pixels
        if written_pixels == 0:
            debug_out["reason"] = "no_destination_pixels_written"
    return True


class UVTEXLINK_State(PropertyGroup):
    has_sample: BoolProperty(default=False)
    last_debug_report: StringProperty(default="")

    image_name: StringProperty(default="")
    image_width: IntProperty(default=0)
    image_height: IntProperty(default=0)
    image_channels: IntProperty(default=0)

    island_count: IntProperty(default=0, min=0)
    clear_old_region: BoolProperty(
        name="Clear Old Region",
        description="Clear each source texture region before writing rescaled result",
        default=False,
    )
    unused_margin_px: IntProperty(
        name="Unused Margin (px)",
        description="Pixel margin around used UV bounds when clearing unused texture space",
        default=4,
        min=0,
        soft_max=64,
    )
    sheet_edge_margin_px: IntProperty(
        name="Sheet Edge Margin (px)",
        description="Protect this many pixels along texture-sheet edges from unused-space clearing",
        default=0,
        min=0,
        soft_max=256,
    )
    bounds_precision_px: FloatProperty(
        name="Bounds Pad (px)",
        description="Expand or shrink source/destination bounds in pixel units for cleaner edge sampling",
        default=0.25,
        min=-8.0,
        max=8.0,
        soft_min=-2.0,
        soft_max=2.0,
        precision=3,
    )
    edge_samples: IntProperty(
        name="Edge Samples",
        description="Sub-samples per axis for each destination texel (higher = cleaner edges, slower)",
        default=2,
        min=1,
        max=6,
    )
    island_transfer_margin_px: IntProperty(
        name="Island Transfer Margin (px)",
        description="Bleed confirmed island colors outward by this many pixels after remap",
        default=0,
        min=0,
        soft_max=64,
    )

    sampled_uv_min_u: FloatProperty(default=0.0)
    sampled_uv_min_v: FloatProperty(default=0.0)
    sampled_uv_max_u: FloatProperty(default=1.0)
    sampled_uv_max_v: FloatProperty(default=1.0)

    # Editable only for single-island samples.
    src_tex_min_u: FloatProperty(default=0.0)
    src_tex_min_v: FloatProperty(default=0.0)
    src_tex_max_u: FloatProperty(default=1.0)
    src_tex_max_v: FloatProperty(default=1.0)


def _set_uv_bounds_to_state(state: UVTEXLINK_State, uv_bounds: Bounds) -> None:
    uv_min_u, uv_min_v, uv_max_u, uv_max_v = uv_bounds
    state.sampled_uv_min_u = uv_min_u
    state.sampled_uv_min_v = uv_min_v
    state.sampled_uv_max_u = uv_max_u
    state.sampled_uv_max_v = uv_max_v


def _set_src_tex_bounds_to_state(state: UVTEXLINK_State, tex_bounds: Bounds) -> None:
    tex_min_u, tex_min_v, tex_max_u, tex_max_v = tex_bounds
    state.src_tex_min_u = tex_min_u
    state.src_tex_min_v = tex_min_v
    state.src_tex_max_u = tex_max_u
    state.src_tex_max_v = tex_max_v


def _get_state_uv_bounds(state: UVTEXLINK_State) -> Bounds:
    return (
        state.sampled_uv_min_u,
        state.sampled_uv_min_v,
        state.sampled_uv_max_u,
        state.sampled_uv_max_v,
    )


def _get_state_tex_bounds(state: UVTEXLINK_State) -> Bounds:
    return (
        state.src_tex_min_u,
        state.src_tex_min_v,
        state.src_tex_max_u,
        state.src_tex_max_v,
    )


class UVTEXLINK_OT_sample_state(Operator):
    bl_idname = "uvtexlink.sample_state"
    bl_label = "Sample UV + Texture"
    bl_description = "Sample selected UV islands and snapshot the active image"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def execute(self, context: bpy.types.Context):
        islands = _selected_uv_islands(context)
        if islands is None:
            self.report({"ERROR"}, "No UV selection found in Edit Mode.")
            return {"CANCELLED"}

        image = _find_active_image(context)
        if image is None:
            self.report({"ERROR"}, "No active image found in Image Editor or active material texture node.")
            return {"CANCELLED"}

        validation_error = _validate_image(image)
        if validation_error:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)
        snapshot = list(image.pixels[:])

        scene_key = context.scene.as_pointer()
        _PIXEL_BUFFERS[scene_key] = {
            "image_name": image.name,
            "width": width,
            "height": height,
            "channels": channels,
            "pixels": snapshot,
            "islands": islands,
        }

        state = context.scene.uvtexlink_state
        state.has_sample = True
        state.image_name = image.name
        state.image_width = width
        state.image_height = height
        state.image_channels = channels
        state.island_count = len(islands)
        state.last_debug_report = ""

        island_bounds = [island["sample_bounds"] for island in islands]
        overall_bounds = _union_bounds(island_bounds)
        if overall_bounds is not None:
            _set_uv_bounds_to_state(state, overall_bounds)

        if len(islands) == 1:
            _set_src_tex_bounds_to_state(state, islands[0]["source_bounds"])

        self.report(
            {"INFO"},
            f"Sampled {len(islands)} UV island(s) and buffered image '{image.name}'.",
        )
        return {"FINISHED"}


class UVTEXLINK_OT_confirm_rescale(Operator):
    bl_idname = "uvtexlink.confirm_rescale"
    bl_label = "Confirm Rescale"
    bl_description = "Rescale buffered texture regions to each island's current UV bounds"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def execute(self, context: bpy.types.Context):
        state = context.scene.uvtexlink_state
        state.last_debug_report = ""
        if not state.has_sample:
            self.report({"ERROR"}, "No sample in buffer. Click 'Sample UV + Texture' first.")
            return {"CANCELLED"}

        scene_key = context.scene.as_pointer()
        buffer = _PIXEL_BUFFERS.get(scene_key)
        if not buffer:
            self.report({"ERROR"}, "Pixel buffer is empty. Re-sample before confirming.")
            state.has_sample = False
            return {"CANCELLED"}

        image = bpy.data.images.get(state.image_name)
        if image is None:
            self.report({"ERROR"}, f"Image '{state.image_name}' no longer exists.")
            return {"CANCELLED"}

        validation_error = _validate_image(image)
        if validation_error:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)

        if (
            width != int(buffer["width"])
            or height != int(buffer["height"])
            or channels != int(buffer["channels"])
        ):
            self.report({"ERROR"}, "Image dimensions/channels changed since sample. Re-sample first.")
            return {"CANCELLED"}

        islands = buffer.get("islands")
        if not islands:
            self.report({"ERROR"}, "No sampled island data found. Re-sample first.")
            return {"CANCELLED"}

        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh in Edit Mode.")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            self.report({"ERROR"}, "Active mesh has no UV layer.")
            return {"CANCELLED"}

        out_pixels = list(image.pixels[:])
        island_transfer_margin = int(state.island_transfer_margin_px)

        for island_index, island in enumerate(islands):
            loop_refs = island.get("loop_refs")
            if not loop_refs:
                self.report({"ERROR"}, f"Sampled island {island_index + 1} has no loop refs.")
                return {"CANCELLED"}
            face_indices = island.get("face_indices")
            sample_triangles = island.get("sample_triangles")
            if not face_indices or not sample_triangles:
                self.report(
                    {"ERROR"},
                    f"Sampled island {island_index + 1} is missing per-pixel island data. Re-sample first.",
                )
                return {"CANCELLED"}

            dst_triangles = _face_uv_triangles_from_face_indices(bm, uv_layer, face_indices)
            if dst_triangles is None or len(dst_triangles) == 0:
                self.report(
                    {"ERROR"},
                    "Mesh topology changed since sample (face refs invalid). Re-sample first.",
                )
                return {"CANCELLED"}
            dst_uv_bounds = _triangles_bounds(dst_triangles)
            if dst_uv_bounds is None:
                self.report(
                    {"ERROR"},
                    "Mesh topology changed since sample (destination island bounds invalid). Re-sample first.",
                )
                return {"CANCELLED"}
            if len(islands) == 1:
                src_tex_bounds = _get_state_tex_bounds(state)
            else:
                src_tex_bounds = island.get("source_bounds")
                if src_tex_bounds is None:
                    self.report({"ERROR"}, f"Island {island_index + 1} is missing source bounds.")
                    return {"CANCELLED"}

            sampled_source_bounds = island.get("source_bounds")
            src_bounds_now = _normalize_bounds(src_tex_bounds)
            if (
                sampled_source_bounds is not None
                and ((src_bounds_now[2] - src_bounds_now[0]) <= _GEOM_EPSILON or (src_bounds_now[3] - src_bounds_now[1]) <= _GEOM_EPSILON)
            ):
                src_tex_bounds = sampled_source_bounds
                if len(islands) == 1:
                    _set_src_tex_bounds_to_state(state, src_tex_bounds)
                self.report(
                    {"WARNING"},
                    f"Island {island_index + 1}: source bounds were zero-area; reverted to sampled source bounds.",
                )

            use_source_mask = True
            src_triangles_for_transform = list(sample_triangles)

            if len(islands) == 1 and sampled_source_bounds is not None:
                if not _bounds_almost_equal(src_tex_bounds, sampled_source_bounds):
                    src_triangles_for_transform = _remap_triangles_between_bounds(
                        sample_triangles,
                        sampled_source_bounds,
                        src_tex_bounds,
                    )
                    use_source_mask = False

            if state.clear_old_region:
                if use_source_mask:
                    _clear_region_masked(
                        out_pixels=out_pixels,
                        width=width,
                        height=height,
                        channels=channels,
                        bounds=src_tex_bounds,
                        triangles=sample_triangles,
                    )
                else:
                    _clear_region(out_pixels, width, height, channels, src_tex_bounds)

            island_mask = bytearray(width * height) if island_transfer_margin > 0 else None
            debug_out: Dict[str, object] = {}
            success = _transform_region_inplace(
                out_pixels=out_pixels,
                source_pixels=buffer["pixels"],
                width=width,
                height=height,
                channels=channels,
                src_tex_bounds=src_tex_bounds,
                dst_tex_bounds=dst_uv_bounds,
                bounds_pad_px=float(state.bounds_precision_px),
                edge_samples=int(state.edge_samples),
                src_island_triangles=src_triangles_for_transform,
                dst_island_triangles=dst_triangles,
                write_mask=island_mask,
                debug_out=debug_out,
            )
            if not success:
                debug_out["island_index"] = island_index + 1
                debug_out["src_bounds_runtime"] = _format_bounds(src_tex_bounds)
                debug_out["dst_bounds_runtime"] = _format_bounds(dst_uv_bounds)
                debug_text = _debug_compact(debug_out)
                state.last_debug_report = debug_text
                print(f"[UVTEXLINK DEBUG] {debug_text}")
                self.report(
                    {"ERROR"},
                    f"Island {island_index + 1} mapping failed. See 'Last Debug Report' in the addon panel.",
                )
                return {"CANCELLED"}

            if island_transfer_margin > 0 and island_mask is not None:
                _bleed_written_pixels(
                    out_pixels=out_pixels,
                    width=width,
                    height=height,
                    channels=channels,
                    written_mask=island_mask,
                    margin_px=island_transfer_margin,
                    bounds=dst_uv_bounds,
                )

        image.pixels = out_pixels
        image.update()

        state.has_sample = False
        state.island_count = 0
        state.last_debug_report = ""
        _PIXEL_BUFFERS.pop(scene_key, None)

        self.report(
            {"INFO"},
            f"Rescaled {len(islands)} sampled island(s) to their current UV bounds.",
        )
        return {"FINISHED"}


class UVTEXLINK_OT_clear_unused_texture_space(Operator):
    bl_idname = "uvtexlink.clear_unused_texture_space"
    bl_label = "Clear Unused Texture Space"
    bl_description = "Clear all texture pixels outside selected UV island regions using configured margins"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def execute(self, context: bpy.types.Context):
        state = context.scene.uvtexlink_state

        image = _find_active_image(context)
        if image is None and state.image_name:
            image = bpy.data.images.get(state.image_name)
        if image is None:
            self.report({"ERROR"}, "No active image found for unused-space cleanup.")
            return {"CANCELLED"}

        validation_error = _validate_image(image)
        if validation_error:
            self.report({"ERROR"}, validation_error)
            return {"CANCELLED"}

        keep_regions = _selected_keep_regions(context)
        if keep_regions is None:
            self.report({"ERROR"}, "Select one or more UV islands to keep before cleanup.")
            return {"CANCELLED"}

        used_bounds, used_triangles = keep_regions
        width = int(image.size[0])
        height = int(image.size[1])
        channels = int(image.channels)

        keep_margin = int(state.unused_margin_px)
        if state.bounds_precision_px > 0.0:
            keep_margin += int(math.ceil(state.bounds_precision_px))
        keep_margin += int(state.island_transfer_margin_px)

        out_pixels = list(image.pixels[:])
        _clear_unused_space(
            out_pixels=out_pixels,
            width=width,
            height=height,
            channels=channels,
            used_bounds=used_bounds,
            used_triangles=used_triangles,
            margin_px=keep_margin,
            sheet_edge_margin_px=int(state.sheet_edge_margin_px),
        )

        image.pixels = out_pixels
        image.update()
        self.report({"INFO"}, f"Cleared unused texture space using {len(used_bounds)} selected island region(s).")
        return {"FINISHED"}


class UVTEXLINK_OT_clear_buffer(Operator):
    bl_idname = "uvtexlink.clear_buffer"
    bl_label = "Clear Buffer"
    bl_description = "Clear sampled UV/texture state and pixel buffer"

    def execute(self, context: bpy.types.Context):
        state = context.scene.uvtexlink_state
        state.has_sample = False
        state.image_name = ""
        state.image_width = 0
        state.image_height = 0
        state.image_channels = 0
        state.island_count = 0
        state.last_debug_report = ""

        scene_key = context.scene.as_pointer()
        _PIXEL_BUFFERS.pop(scene_key, None)

        self.report({"INFO"}, "UV/texture buffer cleared.")
        return {"FINISHED"}


def _draw_uvtexlink_ui(layout: bpy.types.UILayout, context: bpy.types.Context) -> None:
    state = context.scene.uvtexlink_state

    layout.label(text="1) Select UV island(s)")
    layout.label(text="2) Sample")
    layout.label(text="3) Move/scale UV")
    layout.label(text="4) Confirm")

    layout.prop(state, "clear_old_region", text="Clear Old Source Region")
    precision_box = layout.box()
    precision_box.label(text="Bounds Precision")
    precision_box.prop(state, "bounds_precision_px", text="Bounds Pad (px)")
    precision_box.prop(state, "edge_samples", text="Edge Samples")
    precision_box.prop(state, "island_transfer_margin_px", text="Island Transfer Margin (px)")
    cleanup_box = layout.box()
    cleanup_box.label(text="Unused Space Cleanup")
    cleanup_box.prop(state, "unused_margin_px", text="Unused Margin (px)")
    cleanup_box.prop(state, "sheet_edge_margin_px", text="Sheet Edge Margin (px)")

    col = layout.column(align=True)
    col.operator(UVTEXLINK_OT_sample_state.bl_idname, icon="IMPORT")
    col.operator(UVTEXLINK_OT_confirm_rescale.bl_idname, icon="CHECKMARK")
    col.operator(UVTEXLINK_OT_clear_unused_texture_space.bl_idname, icon="BRUSH_DATA")
    col.operator(UVTEXLINK_OT_clear_buffer.bl_idname, icon="TRASH")

    layout.separator()

    if state.last_debug_report:
        debug_box = layout.box()
        debug_box.label(text="Last Debug Report", icon="ERROR")
        for part in state.last_debug_report.split(" || "):
            debug_box.label(text=part[:140])

    if not state.has_sample:
        layout.label(text="Buffer: empty", icon="INFO")
        return

    layout.label(text=f"Image: {state.image_name}", icon="IMAGE_DATA")
    layout.label(text=f"Size: {state.image_width} x {state.image_height}")
    layout.label(text=f"Sampled islands: {state.island_count}")
    layout.label(text=f"Sampled UV: {_format_bounds(_get_state_uv_bounds(state))}")

    if state.island_count == 1:
        box = layout.box()
        box.label(text="Source Texture Bounds (Single Island)")
        row = box.row(align=True)
        row.prop(state, "src_tex_min_u", text="Min U")
        row.prop(state, "src_tex_min_v", text="Min V")
        row = box.row(align=True)
        row.prop(state, "src_tex_max_u", text="Max U")
        row.prop(state, "src_tex_max_v", text="Max V")
        box.label(text="Confirm maps source bounds")
        box.label(text="to this island's current UV bounds")
    else:
        box = layout.box()
        box.label(text="Multi-Island Mode")
        box.label(text="Each island keeps its own sampled")
        box.label(text="source bounds and remaps independently")


class UVTEXLINK_PT_panel_uv_editor(Panel):
    bl_label = "UV Texture Link"
    bl_idname = "UVTEXLINK_PT_panel_uv_editor"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UV Tex Link"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _is_mesh_edit_mode(context)

    def draw(self, context: bpy.types.Context):
        _draw_uvtexlink_ui(self.layout, context)


def _draw_uv_menu(self, context: bpy.types.Context) -> None:
    if not _is_mesh_edit_mode(context):
        return
    layout = self.layout
    layout.separator()
    layout.operator(UVTEXLINK_OT_sample_state.bl_idname, icon="IMPORT")
    layout.operator(UVTEXLINK_OT_confirm_rescale.bl_idname, icon="CHECKMARK")
    layout.operator(UVTEXLINK_OT_clear_unused_texture_space.bl_idname, icon="BRUSH_DATA")
    layout.operator(UVTEXLINK_OT_clear_buffer.bl_idname, icon="TRASH")


classes = (
    UVTEXLINK_State,
    UVTEXLINK_OT_sample_state,
    UVTEXLINK_OT_confirm_rescale,
    UVTEXLINK_OT_clear_unused_texture_space,
    UVTEXLINK_OT_clear_buffer,
    UVTEXLINK_PT_panel_uv_editor,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.uvtexlink_state = PointerProperty(type=UVTEXLINK_State)
    if hasattr(bpy.types, "IMAGE_MT_uvs"):
        bpy.types.IMAGE_MT_uvs.append(_draw_uv_menu)


def unregister():
    if hasattr(bpy.types, "IMAGE_MT_uvs"):
        try:
            bpy.types.IMAGE_MT_uvs.remove(_draw_uv_menu)
        except ValueError:
            pass
    if hasattr(bpy.types.Scene, "uvtexlink_state"):
        del bpy.types.Scene.uvtexlink_state
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
