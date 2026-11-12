import os

import bpy
from mathutils import Vector


device = os.environ.get("ZLUDA_TEST_DEVICE", "CPU").upper()
output = os.environ["ZLUDA_TEST_OUTPUT"]
use_custom_normals = os.environ.get("ZLUDA_TEST_CUSTOM_NORMALS", "1") == "1"
use_true_normal = os.environ.get("ZLUDA_TEST_TRUE_NORMAL", "0") == "1"
normal_pattern = os.environ.get("ZLUDA_TEST_NORMAL_PATTERN", "gradient").lower()
sphere_segments = int(os.environ.get("ZLUDA_TEST_SPHERE_SEGMENTS", "48"))
sphere_ring_count = int(os.environ.get("ZLUDA_TEST_SPHERE_RING_COUNT", "24"))
if device not in {"CPU", "GPU"}:
    raise RuntimeError(f"Unsupported test device: {device}")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = device
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = output
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"

if normal_pattern in {"triangle_constants", "triangle_gradient", "triangle_shared_constants"}:
    shared_vertices = normal_pattern == "triangle_shared_constants"
    mesh = bpy.data.meshes.new("Packed normal triangle probe")
    vertices = [(-1.5, -0.9, 0.0), (-0.1, -0.9, 0.0), (-0.8, 0.9, 0.0)]
    if shared_vertices:
        vertices.append((1.5, -0.9, 0.0))
        triangles = [(0, 1, 2), (1, 3, 2)]
    else:
        vertices.extend([(0.1, -0.9, 0.0), (1.5, -0.9, 0.0), (0.8, 0.9, 0.0)])
        triangles = [(0, 1, 2), (3, 4, 5)]
    mesh.from_pydata(vertices, [], triangles)
    sphere = bpy.data.objects.new("Packed normal probe", mesh)
    scene.collection.objects.link(sphere)
    sphere.select_set(True)
    bpy.context.view_layer.objects.active = sphere
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if use_custom_normals:
        triangle_normals = (
            [(0.0, 0.0, 1.0)] * 3 + [(1.0, 0.0, 0.0)] * 3
            if normal_pattern in {"triangle_constants", "triangle_shared_constants"}
            else [
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        )
        mesh.normals_split_custom_set(triangle_normals)
else:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=sphere_segments, ring_count=sphere_ring_count, radius=1.0
    )
    sphere = bpy.context.object
    sphere.name = "Packed normal probe"
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True

    if normal_pattern == "sphere_face_constants":
        axes = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        custom_normals = [None] * len(sphere.data.loops)
        for polygon in sphere.data.polygons:
            normal = axes[polygon.index % len(axes)]
            for loop_index in polygon.loop_indices:
                custom_normals[loop_index] = normal
    else:
        custom_normals = []
        for vertex in sphere.data.vertices:
            p = vertex.co.normalized()
            if normal_pattern == "constant_z":
                normal = Vector((0.0, 0.0, 1.0))
            elif normal_pattern == "constant_x":
                normal = Vector((1.0, 0.0, 0.0))
            elif normal_pattern == "gradient":
                normal = Vector((0.8 * p.x + 0.1, 0.8 * p.y - 0.2, 0.8 * p.z + 0.25))
            else:
                raise RuntimeError(f"Unsupported normal pattern: {normal_pattern}")
            custom_normals.append(normal.normalized())
    if use_custom_normals:
        if normal_pattern == "sphere_face_constants":
            sphere.data.normals_split_custom_set(custom_normals)
        else:
            sphere.data.normals_split_custom_set_from_vertices(custom_normals)

material = bpy.data.materials.new("Packed normal visualization")
material.use_nodes = True
nodes = material.node_tree.nodes
nodes.clear()
geometry = nodes.new("ShaderNodeNewGeometry")
scale = nodes.new("ShaderNodeVectorMath")
scale.operation = "SCALE"
scale.inputs[3].default_value = 0.5
normal_socket = geometry.outputs["True Normal" if use_true_normal else "Normal"]
material.node_tree.links.new(normal_socket, scale.inputs[0])
offset = nodes.new("ShaderNodeVectorMath")
offset.operation = "ADD"
offset.inputs[1].default_value = (0.5, 0.5, 0.5)
material.node_tree.links.new(scale.outputs["Vector"], offset.inputs[0])
emission = nodes.new("ShaderNodeEmission")
material.node_tree.links.new(offset.outputs["Vector"], emission.inputs["Color"])
output_node = nodes.new("ShaderNodeOutputMaterial")
material.node_tree.links.new(emission.outputs["Emission"], output_node.inputs["Surface"])
sphere.data.materials.append(material)

bpy.ops.object.camera_add(location=(0.0, 0.0, 4.0))
camera = bpy.context.object
camera.name = "Probe camera"
camera.data.type = "ORTHO"
camera.data.ortho_scale = (
    3.4 if normal_pattern in {"triangle_constants", "triangle_shared_constants"} else 2.5
)
scene.camera = camera
camera.rotation_euler = (0.0, 0.0, 0.0)

if device == "GPU":
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    selected = [
        item
        for item in preferences.devices
        if item.type == "CUDA" and "ZLUDA" in item.name.upper()
    ]
    for item in preferences.devices:
        item.use = item in selected
        print(
            f"PACKED_NORMAL_DEVICE|type={item.type}|"
            f"name={item.name!r}|use={item.use}",
            flush=True,
        )
    if not selected:
        raise RuntimeError("No ZLUDA CUDA device found; refusing CPU fallback")

print(
    f"PACKED_NORMAL_CONFIG|device={device}|resolution=256x256|"
    f"samples={scene.cycles.samples}|custom_normals={sphere.data.has_custom_normals}|"
    f"true_normal={use_true_normal}|pattern={normal_pattern}",
    flush=True,
)
bpy.ops.render.render(write_still=True)
if not os.path.isfile(output) or os.path.getsize(output) < 1024:
    raise RuntimeError(f"Render output missing or too small: {output}")
print(f"PACKED_NORMAL_COMPLETE|device={device}|output={output}", flush=True)
