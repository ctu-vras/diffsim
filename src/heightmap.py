import warp as wp

@wp.struct
class Heightmap:
    heights: wp.array2d(dtype=wp.float32)
    ke: wp.array2d(dtype=wp.float32)
    kd: wp.array2d(dtype=wp.float32)
    kf: wp.array2d(dtype=wp.float32)
    origin: wp.vec3
    resolution: wp.float32
    width: wp.int32
    length: wp.int32

@wp.kernel
def eval_heightmap_collisions_shoot(height_map_array: wp.array(dtype=Heightmap),
                                    body_q: wp.array2d(dtype=wp.transformf),
                                    body_qd: wp.array2d(dtype=wp.spatial_vectorf),
                                    sim_idx: int,
                                    T_s: int,
                                    track_velocities: wp.array3d(dtype=wp.float32),
                                    contact_points: wp.array3d(dtype=wp.vec3),
                                    constraint_forces: wp.array2d(dtype=wp.vec3),
                                    friction_forces: wp.array2d(dtype=wp.vec3),
                                    collisions: wp.array2d(dtype=wp.vec3),
                                    body_f: wp.array2d(dtype=wp.spatial_vectorf)):
    shoot_idx, robot_idx, contact_idx = wp.tid()

    # parse heightmap data
    height_map = height_map_array[robot_idx]
    heights = height_map.heights
    kes = height_map.ke
    kds = height_map.kd
    kfs = height_map.kf
    hm_origin = height_map.origin
    hm_res = height_map.resolution
    width = height_map.width
    length = height_map.length

    # compute current shoot simulation timestep
    current_state_idx = shoot_idx * T_s + sim_idx
    current_force_idx = (T_s - 1) * shoot_idx + sim_idx

    # acess simulation state
    robot_to_world = body_q[current_state_idx, robot_idx]
    robot_to_world_speed = body_qd[current_state_idx, robot_idx]
    wheel_to_robot_pos = contact_points[shoot_idx, robot_idx, contact_idx]

    # transform contact state to heightmap frame
    forward_to_world = wp.transform_vector(robot_to_world, wp.vec3(1.0, 0.0, 0.0))
    wheel_to_world_pos = wp.transform_point(robot_to_world, wheel_to_robot_pos)
    wheel_to_world_vel = wp.cross(wp.spatial_top(robot_to_world_speed), wheel_to_robot_pos) + wp.spatial_bottom(
        robot_to_world_speed)
    wheel_to_hm = wheel_to_world_pos - hm_origin
    # x, y normalized by the heightmap resolution
    x_n = wheel_to_hm[0] / hm_res
    y_n = wheel_to_hm[1] / hm_res
    # cell index
    u = wp.int(wp.floor(x_n))
    v = wp.int(wp.floor(y_n))

    if u < 0 or u >= width or v < 0 or v >= length:  # cell outside heightmap
        return

    # hm_height = hm[x_id, z_id]  # simpler version without interpolation

    # relative position of the wheel inside the cell
    x_r = x_n - wp.float(u)
    y_r = y_n - wp.float(v)

    # useful terms for height and terrain normal
    a = heights[u, v]
    b = heights[u + 1, v]
    c = heights[u, v + 1]
    d = heights[u + 1, v + 1]

    adbc = a + d - b - c
    ba = b - a
    ca = c - a

    hm_height = x_r * y_r * adbc + x_r * ba + y_r * ca + a

    wheel_height = wheel_to_hm[2]
    # This particular wheel is above ground, so no collision force is generated
    if wheel_height > hm_height:
        return

    # compute normal to the terrain at the horizontal position of the wheel
    n = wp.vec3(-y_r * adbc - ba, -x_r * adbc - ca, 1.0)
    n = wp.normalize(n)

    # depth of penetration
    d = hm_height - wheel_height

    # normal and tangential velocity components
    v_n = wp.dot(wheel_to_world_vel, n)
    v_t = wheel_to_world_vel - n * v_n

    # compute the track velocity at the wheel position
    tangential_track_direction = forward_to_world - n*wp.dot(forward_to_world, n)
    tangential_track_velocity = wp.vec3(0.0, 0.0, 0.0)
    if wp.length(tangential_track_direction) > 1e-4:
        vel_idx = (contact_idx % 2)
        track_vel = track_velocities[current_state_idx, robot_idx, vel_idx]  # left and right track velocities
        tangential_track_velocity = wp.normalize(tangential_track_direction) * track_vel

    # compute the constraint (penetration force) and friction force
    constraint_force = n * (kes[u, v] * d - kds[u, v] * v_n)
    friction_force = -kfs[u, v] * (v_t - tangential_track_velocity) * wp.length(constraint_force)
    total_force = constraint_force + friction_force

    # combine into wrench and add to total robot force
    robot_wrench = wp.spatial_vector(wp.cross(wheel_to_robot_pos, total_force), total_force)
    wp.atomic_add(body_f, current_force_idx, robot_idx, robot_wrench)

    if robot_idx != 0:
        return

    # Store the contact info only for the first robot TODO: vis all robots?
    constraint_forces[current_state_idx, contact_idx] = constraint_force
    friction_forces[current_state_idx, contact_idx] = friction_force
    collisions[current_state_idx, contact_idx] = wp.vec3(wheel_to_world_pos[0], wheel_to_world_pos[1], hm_height + hm_origin[2])