"""Procedural direct-light scenes; bpy is imported only inside the renderer process."""


def material(bpy, name, color, *, emission=False, checker=False, spec=None):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes = result.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeEmission" if emission else "ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (*color, 1)
    result.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
    if checker:
        spec = spec or {}
        pattern = nodes.new("ShaderNodeTexChecker")
        pattern.inputs["Color1"].default_value = (*spec.get("floor_color1", [0.38, 0.42, 0.46]), 1)
        pattern.inputs["Color2"].default_value = (*spec.get("floor_color2", [0.65, 0.60, 0.50]), 1)
        pattern.inputs["Scale"].default_value = spec.get("checker_scale", 12)
        result.node_tree.links.new(pattern.outputs["Color"], shader.inputs["Color"])
    return result


def build_scene(config: dict, spec: dict):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = config["renderer"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = render["samples"]
    scene.cycles.seed = render["seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 1
    for key in ("diffuse_bounces", "glossy_bounces", "transmission_bounces", "volume_bounces"):
        setattr(scene.cycles, key, 0)
    scene.render.threads_mode = "FIXED"
    scene.render.threads = render["threads"]
    scene.render.resolution_x = scene.render.resolution_y = render["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("black_environment")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
    scene.world = world
    bpy.ops.mesh.primitive_plane_add(size=12)
    floor = bpy.context.object
    floor.name = "background"
    floor.data.materials.append(
        material(bpy, "checker_floor", (0.5, 0.5, 0.5), checker=True, spec=spec)
    )
    width = spec["object_width"]
    location = (*spec.get("source_xy", config["source_xy"]), width / 2)
    if spec["shape"] == "cube":
        bpy.ops.mesh.primitive_cube_add(size=width, location=location)
    else:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48, radius=width / 2, depth=width, location=location
        )
    obj = bpy.context.object
    obj.name = "translated_object"
    obj.pass_index = 1
    obj.data.materials.append(material(bpy, "object_diffuse", spec["color"]))
    bpy.ops.object.light_add(type="POINT", location=spec["light_position"])
    light = bpy.context.object
    light.data.energy = spec.get("light_energy", config["light_energy"])
    light.data.shadow_soft_size = 0
    bpy.ops.object.camera_add(location=config["camera_position"])
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector(config["camera_target"]) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = config["ortho_scale"]
    scene.camera = camera
    return scene, obj, floor, light
