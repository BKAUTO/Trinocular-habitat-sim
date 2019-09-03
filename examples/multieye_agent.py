# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a demonstration of how to create an agent with
two cameras in a stereo pair.


This can be done by giving the agent two sensors (be it RGB, depth, or semantic)
with different positions.

Note that the cameras must have different uuid's
"""

import random, IPython, time, cv2

import numpy as np

import habitat_sim

from habitat_sim.agent import controls
from habitat_sim.agent.controls import ActuationSpec

from habitat_sim.agent import default_controls


# Helper function to render observations from the stereo agent
def render(sim, BF, display=True):
    for _ in range(10000):
        Record = False
        key = cv2.waitKeyEx(0)
        print("key:", key)

        if key == ord("q"):
            break
        elif key == ord('w'):
            obs = sim.step("move_forward")
        elif key == ord('s'):
            obs = sim.step("move_backward")
        elif key == ord('a'):
            obs = sim.step("move_left")
        elif key == ord('d'):
            obs = sim.step("move_right")
        elif key == ord('z'):
            obs = sim.step("move_up")
        elif key == ord('x'):
            obs = sim.step("move_down")
        elif key == ord('i'): #65362:
            obs = sim.step("turn_up")
        elif key == ord('k'): #65364:
            obs = sim.step("turn_down")
        elif key == ord('j'): #65361:
            obs = sim.step("turn_left")
        elif key == ord('l'): #65363:
            obs = sim.step("turn_right")
        elif key == ord('r'):
            obs = sim.get_sensor_observations()
            Record = True
        else:
            print('unknown key: ', key)
            continue

        # print(sim.get_agent(0).state)

        rgb0 = obs["left_rgb"]
        rgb1 = obs["center_rgb"]
        rgb2 = obs["right_rgb"]
        rgb3 = obs["bottom_rgb"]
        rgb4 = obs["top_rgb"]
        depth0 = obs["left_depth"]
        depth1 = obs["center_depth"]
        depth2 = obs["right_depth"]
        depth3 = obs["bottom_depth"]
        depth4 = obs["top_depth"]
        disparity0 = Baseline * fx / depth0
        disparity1 = Baseline * fx / depth1
        disparity2 = Baseline * fx / depth2
        disparity3 = Baseline * fx / depth3
        disparity4 = Baseline * fx / depth4

        # # If it is a depth pair, manually normalize into [0, 1]
        # # so that images are always consistent
        # depth_pair = np.clip(depth_pair, 0, 10)
        # depth_pair /= 10.0

        # If in RGB/RGBA format, change first to RGB and change to BGR
        rgb0 = rgb0[..., 0:3][..., ::-1]
        rgb1 = rgb1[..., 0:3][..., ::-1]
        rgb2 = rgb2[..., 0:3][..., ::-1]
        rgb3 = rgb3[..., 0:3][..., ::-1]
        rgb4 = rgb4[..., 0:3][..., ::-1]

        # disparity map
        disparity0[disparity0>255] = 0
        disparity1[disparity1>255] = 0
        disparity2[disparity2>255] = 0
        disparity3[disparity3>255] = 0
        disparity4[disparity4>255] = 0

        # display=False is used for the smoke test
        if display:
            cv2.imshow("stereo_pair", np.concatenate([cv2.resize(rgb0, (0,0), fx=0.5, fy=0.5),
                                                      cv2.resize(rgb1, (0,0), fx=0.5, fy=0.5), 
                                                      cv2.resize(rgb2, (0,0), fx=0.5, fy=0.5)], axis=1))
            # cv2.imshow("depth_pair", np.concatenate([disparity1/255.0*4, disparity2/255.0*4], axis=1) )

        if Record == True:
            Enlarge = 1
            cv2.imwrite("view0.png", rgb0)
            cv2.imwrite("view1.png", rgb1)
            cv2.imwrite("view2.png", rgb2)
            cv2.imwrite("view3.png", rgb3)
            cv2.imwrite("view4.png", rgb4)
            cv2.imwrite("disp0.png", disparity0*Enlarge)
            cv2.imwrite("disp1.png", disparity1*Enlarge)
            cv2.imwrite("disp2.png", disparity2*Enlarge)
            cv2.imwrite("disp3.png", disparity3*Enlarge)
            cv2.imwrite("disp4.png", disparity4*Enlarge)



if __name__ == "__main__":
    
    STEP = 0.1
    Width = 1280
    Height = 1080
    Baseline = 0.2
    Cam_H = 1.5
    HFOV = np.pi / 2.0
    fx = 1 / np.tan(HFOV / 2.0) * Width / 2.0
    BF = Baseline * fx

    cv2.namedWindow("stereo_pair")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene.id = (
        "/home/wbk/Desktop/gibson/Wyatt.glb"
    )

    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.SensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_rgb"
    left_rgb_sensor.resolution = [Height, Width]
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    left_rgb_sensor.position = Cam_H * habitat_sim.geo.UP + Baseline * habitat_sim.geo.LEFT

    right_rgb_sensor = habitat_sim.SensorSpec()
    right_rgb_sensor.uuid = "right_rgb"
    right_rgb_sensor.resolution = [Height, Width]
    right_rgb_sensor.position = Cam_H * habitat_sim.geo.UP + Baseline * habitat_sim.geo.RIGHT

    center_rgb_sensor = habitat_sim.SensorSpec()
    center_rgb_sensor.uuid = "center_rgb"
    center_rgb_sensor.resolution = [Height, Width]
    center_rgb_sensor.position = Cam_H * habitat_sim.geo.UP

    bottom_rgb_sensor = habitat_sim.SensorSpec()
    bottom_rgb_sensor.uuid = "bottom_rgb"
    bottom_rgb_sensor.resolution = [Height, Width]
    bottom_rgb_sensor.position = (Cam_H - Baseline) * habitat_sim.geo.UP

    top_rgb_sensor = habitat_sim.SensorSpec()
    top_rgb_sensor.uuid = "top_rgb"
    top_rgb_sensor.resolution = [Height, Width]
    top_rgb_sensor.position = (Cam_H + Baseline) * habitat_sim.geo.UP

    # Now let's do the exact same thing but for a depth camera stereo pair!
    left_depth_sensor = habitat_sim.SensorSpec()
    left_depth_sensor.uuid = "left_depth"
    left_depth_sensor.resolution = [Height, Width]
    left_depth_sensor.position = Cam_H * habitat_sim.geo.UP + Baseline * habitat_sim.geo.LEFT
    # The only difference is that we set the sensor type to DEPTH
    left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    right_depth_sensor = habitat_sim.SensorSpec()
    right_depth_sensor.uuid = "right_depth"
    right_depth_sensor.resolution = [Height, Width]
    right_depth_sensor.position = Cam_H * habitat_sim.geo.UP + Baseline * habitat_sim.geo.RIGHT
    right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    center_depth_sensor = habitat_sim.SensorSpec()
    center_depth_sensor.uuid = "center_depth"
    center_depth_sensor.resolution = [Height, Width]
    center_depth_sensor.position = Cam_H * habitat_sim.geo.UP
    center_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    bottom_depth_sensor = habitat_sim.SensorSpec()
    bottom_depth_sensor.uuid = "bottom_depth"
    bottom_depth_sensor.resolution = [Height, Width]
    bottom_depth_sensor.position = (Cam_H - Baseline) * habitat_sim.geo.UP
    bottom_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    top_depth_sensor = habitat_sim.SensorSpec()
    top_depth_sensor.uuid = "top_depth"
    top_depth_sensor.resolution = [Height, Width]
    top_depth_sensor.position = (Cam_H + Baseline) * habitat_sim.geo.UP
    top_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    
    agent_config = habitat_sim.AgentConfiguration()
    # ====================== add actions =========================
    habitat_sim.controls.register_move_fn(default_controls.LookLeft, name="cam_look_left", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.LookRight, name="cam_look_right", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.LookUp, name="cam_look_up", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.LookDown, name="cam_look_down", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.MoveForward, name="cam_move_forward", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.MoveBackward, name="cam_move_backward", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.MoveLeft, name="cam_move_left", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.MoveRight, name="cam_move_right", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.MoveUp, name="cam_move_up", body_action=False)
    habitat_sim.controls.register_move_fn(default_controls.MoveDown, name="cam_move_down", body_action=False)
    
    print(agent_config.action_space)

    agent_config.action_space["move_forward"] = habitat_sim.ActionSpec(
        "move_forward", ActuationSpec(amount=STEP*2)
    )
    agent_config.action_space["move_backward"] = habitat_sim.ActionSpec(
        "move_backward", ActuationSpec(amount=STEP)
    )
    agent_config.action_space["move_left"] = habitat_sim.ActionSpec(
        "move_left", ActuationSpec(amount=STEP)
    )
    agent_config.action_space["move_right"] = habitat_sim.ActionSpec(
        "move_right", ActuationSpec(amount=STEP)
    )
    agent_config.action_space["move_up"] = habitat_sim.ActionSpec(
        "move_up", ActuationSpec(amount=STEP)
    )
    agent_config.action_space["move_down"] = habitat_sim.ActionSpec(
        "move_down", ActuationSpec(amount=STEP)
    )
    agent_config.action_space["turn_up"] = habitat_sim.ActionSpec(
        "look_up", ActuationSpec(amount=5.0)
    )
    agent_config.action_space["turn_down"] = habitat_sim.ActionSpec(
        "look_down", ActuationSpec(amount=5.0)
    )
    agent_config.action_space["turn_left"] = habitat_sim.ActionSpec(
        "turn_left", ActuationSpec(amount=5.0)
    )
    agent_config.action_space["turn_right"] = habitat_sim.ActionSpec(
        "turn_right", ActuationSpec(amount=5.0)
    )
    print(agent_config.action_space)

    # ============================================================

    # Now we simly set the agent's list of sensor specs to be the two specs for our two sensors
    agent_config.sensor_specifications = [left_rgb_sensor, center_rgb_sensor, right_rgb_sensor, bottom_rgb_sensor, top_rgb_sensor,
                        left_depth_sensor, center_depth_sensor, right_depth_sensor, bottom_depth_sensor, top_depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    render(sim, BF)
    sim.close()
    del sim

    # # Now let's do the exact same thing but for a depth camera stereo pair!
    # left_depth_sensor = habitat_sim.SensorSpec()
    # left_depth_sensor.uuid = "left_sensor"
    # left_depth_sensor.resolution = [1280, 1080]
    # left_depth_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
    # # The only difference is that we set the sensor type to DEPTH
    # left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    # right_depth_sensor = habitat_sim.SensorSpec()
    # right_depth_sensor.uuid = "right_sensor"
    # right_depth_sensor.resolution = [1280, 1080]
    # right_depth_sensor.position = (
    #     1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT
    # )
    # # The only difference is that we set the sensor type to DEPTH
    # right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    # agent_config = habitat_sim.AgentConfiguration()
    # agent_config.sensor_specifications = [left_depth_sensor, right_depth_sensor]

    # sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    # _render(sim, display, depth=True)
    # sim.close()
    # del sim
