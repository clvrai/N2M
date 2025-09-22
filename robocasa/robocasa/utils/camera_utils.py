"""
Collection of constants for cameras / robots / etc
in kitchen environments
"""

# default free cameras for different kitchen layouts
LAYOUT_CAMS = {
    0: dict(
        lookat=[2.26593463, -1.00037131, 1.38769295],
        distance=3.0505089839567323,
        azimuth=90.71563812375285,
        elevation=-12.63948837207208,
    ),
    1: dict(
        lookat=[2.66147999, -1.00162429, 1.2425155],
        distance=3.7958766287746255,
        azimuth=89.75784013699234,
        elevation=-15.177406642875091,
    ),
    2: dict(
        lookat=[3.02344359, -1.48874618, 1.2412914],
        distance=3.6684844368165512,
        azimuth=51.67880851867874,
        elevation=-13.302619131542388,
    ),
    # 3: dict(
    #     lookat=[11.44842548, -11.47664723, 11.24115989],
    #     distance=43.923271794728187,
    #     azimuth=227.12928449329333,
    #     elevation=-16.495686334624907,
    # ),
    4: dict(
        lookat=[1.6, -1.0, 1.0],
        distance=5,
        azimuth=89.70301806083651,
        elevation=-18.02177994296577,
    ),
}

DEFAULT_LAYOUT_CAM = {
    "lookat": [2.25, -1, 1.05312667],
    "distance": 5,
    "azimuth": 89.70301806083651,
    "elevation": -18.02177994296577,
}

CAM_CONFIGS = dict(
    DEFAULT=dict(
        robot0_agentview_center=dict(
            pos=[-0.6, 0.0, 1.15],
            quat=[
                0.636945903301239,
                0.3325185477733612,
                -0.3199238181114197,
                -0.6175596117973328,
            ],
            parent_body="mobilebase0_support",
        ),
        robot0_agentview_left=dict(
            pos=[-0.5, 0.35, 1.05],
            quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_agentview_right=dict(
            pos=[-0.5, -0.35, 1.05],
            quat=[
                0.6775091886520386,
                0.3767866790294647,
                -0.2993525564670563,
                -0.55623859167099,
            ],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_frontview=dict(
            pos=[-0.50, 0, 0.95],
            quat=[
                0.6088936924934387,
                0.3814677894115448,
                -0.3673907518386841,
                -0.5905545353889465,
            ],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_eye_in_hand=dict(
            pos=[0.05, 0, 0],
            quat=[0, 0.707107, 0.707107, 0],
            parent_body="robot0_right_hand",
        ),
        
        # newly added for depth camera
        robot0_front_depth=dict(
            pos=[0.1, 0, 0.05],
            quat=[0, 0.707107, 0.707107, 0],
            camera_attribs=dict(fovy="100"),
            parent_body="robot0_right_hand",
        ),
    ),
    ### Add robot specific configs here ####
    PandaMobile=dict(),
    GR1FixedLowerBody=dict(),
)
    
    # quat follows [w, x, y, z] convention
CAM_CONFIGS_FOR_LAYOUT = {
    0: dict(
        depth_camera1 = dict(
            pos =[1.31081524,-1.5, 2.0],  # Position the camera above the scene
            quat= [0.89085329,0.45428215,0.00172727,0.00226997],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera2 = dict(
            pos = [2.77269084, -1.5, 2.0],  # Position the camera above the scene
            quat= [0.89085303, 0.45428195, 0.00177944, 0.00237227],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera3 = dict(
            pos =[4.33561248, -1.5, 2.0],  # Position the camera above the scene
            quat= [0.89085273, 0.45428173, 0.00183503, 0.0024813],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera4 = dict(
            pos =[0.55661682, -2.30373199, 1.63387535],  # Position the camera above the scene
            quat= [0.82848821, 0.42269513, -0.16651527, -0.32742751],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera5 = dict(
            pos =[4.95445254, -2.39008951, 1.63377875],  # Position the camera above the scene
            quat= [0.81444214, 0.41524064, 0.18457086, 0.36082242],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    1: dict(
        depth_camera1 = dict(
            pos =[0.8,-1.7, 2.1],  # Position the camera above the scene
            quat= [0.89085329,0.45428215,0.00172727,0.00226997],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera2 = dict(
            pos = [2.4, -1.7, 2.1],  # Position the camera above the scene
            quat= [0.89085303, 0.45428195, 0.00177944, 0.00237227],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera3 = dict(
            pos =[4.0, -1.7, 2.1],  # Position the camera above the scene
            quat= [0.89085273, 0.45428173, 0.00183503, 0.0024813],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera4 = dict(
            pos =[-0.44732172, -3.41286694, 2.0],  # Position the camera above the scene
            quat= [0.83904097, 0.42805405, -0.15213539, -0.29939072],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera5 = dict(
            pos =[4.77014049, -3.48505517, 2.0],  # Position the camera above the scene
            quat= [0.82931622, 0.42269541, 0.16644529, 0.32536003],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    
    
    2: dict(
        depth_camera1 = dict(
            pos =[1.86558718, -1.6, 2.0],  # Position the camera above the scene
            quat= [0.89085329, 0.45428215, 0.00172727, 0.00226997],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera2 = dict(
            pos = [4.0, -1.6, 2.0],  # Position the camera above the scene
            quat= [0.89084522, 0.45429783, 0.00174047, 0.00229069],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera3 = dict(
            pos =[3.6, -1.2, 2.0],  # Position the camera above the scene
            quat= [-0.62986086, -0.32146625, 0.32078995, 0.63010202],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera4 = dict(
            pos =[3.6, -2.95860774, 2.0],  # Position the camera above the scene
            quat= [0.63037141, 0.32157797, -0.3206569, -0.62960197],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera5 = dict(
            pos =[1.07098718, -4.06240355, 2.5],  # Position the camera above the scene
            quat= [0.8467968, 0.37248128, -0.15319281, -0.34745479],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
),
    
    3: dict(
        depth_camera1 = dict(
            pos =[1.55253347, -3.3988027, 2.0],  # Position the camera above the scene
            quat= [0.63120505, 0.32137672, 0.32081341, 0.62878927],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera2 = dict(
            pos = [1.71436985, -1.27664171, 2.0],  # Position the camera above the scene
            quat= [0.57524533, 0.29289305, 0.3471033, 0.68031299],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera3 = dict(
            pos =[1.20234391, -1.44110366, 2.0],  # Position the camera above the scene
            quat= [8.90880611e-01, 4.54236911e-01, 2.63410535e-04, -7.04825383e-04],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera4 = dict(
            pos =[3.07206719, -1.44963655, 2.0],  # Position the camera above the scene
            quat= [0.88785514, 0.45289602, 0.03731502, 0.07215279],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera5 = dict(
            pos =[3.78361928, -4.67117282, 2.50],  # Position the camera above the scene
            quat= [0.88090621, 0.31217491, 0.11939474, 0.3351059],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    
    4: dict(
        depth_camera1 = dict(
            pos =[1.63767722, -0.91852038, 2.0],  # Position the camera above the scene
            quat= [0.63171454, 0.32257084, -0.3200607, -0.62804931],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera2 = dict(
            pos = [1.71059568, -3.1543501, 2.0],  # Position the camera above the scene
            quat= [0.631971, 0.32261027, -0.31965376, -0.6279783],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera3 = dict(
            pos =[1.78567421, -1.68552179, 2.0],  # Position the camera above the scene
            quat= [0.66308323, 0.33772934, 0.30357824, 0.59506283],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera4 = dict(
            pos =[1.60934593, -4.42627126, 2.0],  # Position the camera above the scene
            quat= [0.78345579, 0.39898068, 0.21681325, 0.42426815],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera5 = dict(
            pos =[1.40998332, -4.29915272, 2.0],  # Position the camera above the scene
            quat= [0.79152467, 0.40374747, -0.20794238, -0.40894578],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    
    5: dict(
        depth_camera1 = dict(
            pos =[1.36264049, -3.10899139, 2.0],  # Position the camera above the scene
            quat= [0.74092425, 0.37774458, 0.25261909, 0.49449356],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera2 = dict(
            pos = [1.63959303, -1.57017889, 2.0],  # Position the camera above the scene
            quat= [0.68173621, 0.34763235, 0.29279868, 0.57328564],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera3 = dict(
            pos =[3.31825024, -1.53821522, 2.0],  # Position the camera above the scene
            quat= [0.87252307, 0.44461412, 0.09228059, 0.18029436],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera4 = dict(
            pos =[2.99537322, -1.55724981, 2.0],  # Position the camera above the scene
            quat= [0.82848821, 0.42269513, -0.16651527, -0.32742751],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera5 = dict(
            pos =[1.21626949, -3.06652611, 2.5],  # Position the camera above the scene
            quat= [0.82691434, 0.4078476, -0.16966524, -0.34797517],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    
    6: dict(
        depth_camera1 = dict(
            pos =[1.49401365, -3.70250102, 2.0],  # Position the camera above the scene
            quat= [0.75181303, 0.38334696, 0.24410157, 0.47774125],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera2 = dict(
            pos = [1.74952032, -1.13572676, 2.0],  # Position the camera above the scene
            quat= [0.53947594, 0.27477311, 0.36174586, 0.70894667],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera3 = dict(
            pos =[1.03829179, -1.32406004, 2.0],  # Position the camera above the scene
            quat= [0.8773363, 0.44756804, -0.07841675, -0.15432003],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera4 = dict(
            pos =[5.21809278, -1.80180553, 2.0],  # Position the camera above the scene
            quat= [0.8659777, 0.44132514, 0.10716716, 0.20935601],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera5 = dict(
            pos =[4.5317078, -1.76195871, 1.9],  # Position the camera above the scene
            quat= [0.64258847, 0.30292627, -0.32124852, -0.62619096],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    
    7: dict(
        depth_camera1 = dict(
            pos =[1.68106629, -1.68539208, 2.0],  # Position the camera above the scene
            quat= [0.63034087, 0.32129843, 0.32138271, 0.62940517],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera2 = dict(
            pos = [1.02081482, -1.39240755, 2.0],  # Position the camera above the scene
            quat= [0.88803102, 0.45263155, -0.03617298, -0.07222948],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera3 = dict(
            pos =[3.33129439, -1.45878895, 2.0],  # Position the camera above the scene
            quat= [0.88261481, 0.45025206, 0.06188773, 0.12014194],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera4 = dict(
            pos =[3.06783822, -1.2701851, 2.0],  # Position the camera above the scene
            quat= [-0.61118431, -0.31204916, 0.33036335, 0.6480271],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera5 = dict(
            pos =[2.48422159, -4.17135696, 1.65476038],  # Position the camera above the scene
            quat= [0.85211872, 0.52172044, -0.02048712, -0.03580144],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),

    8: dict(
        depth_camera1 = dict(
            pos =[4.20995252, -3.01838048, 2.0],  # Position the camera above the scene
            quat= [0.63725815, 0.32561272, -0.31711075, -0.62234972],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera2 = dict(
            pos = [3.62507374, -1.69103306, 2.0],  # Position the camera above the scene
            quat= [0.88829378, 0.45329375, -0.03331602, -0.06594674],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),

        depth_camera3 = dict(
            pos =[2.60150616, -1.36219323, 2.0],  # Position the camera above the scene
            quat= [0.89073517, 0.4544993, 0.00255391, 0.00383564],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera4 = dict(
            pos =[1.7125589, -1.75634295, 2.0],  # Position the camera above the scene
            quat= [0.87977957, 0.4488039, 0.0716746, 0.13937622],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        
        depth_camera5 = dict(
            pos =[1.46203005, -1.63491261, 2.0],  # Position the camera above the scene
            quat= [0.60155816, 0.30631319, 0.33523268, 0.65720549],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
    
    9: dict(
        depth_camera1 = dict(
            pos =[4.39583423, -1.62374295, 2.0],  # Position the camera above the scene
            quat= [0.66633316, 0.33985786, -0.30115108, -0.59144296],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera2 = dict(
            pos = [3.47424318, -1.22956736, 2.0],  # Position the camera above the scene
            quat= [0.85578069, 0.43674075, -0.12574922, -0.24715191],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera3 = dict(
            pos =[1.62992233, -1.66224761, 2.0],  # Position the camera above the scene
            quat= [0.89092102, 0.45414599, 0.00180755, 0.00280748],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera4 = dict(
            pos =[0.88278781, -1.34774258, 2.0],  # Position the camera above the scene
            quat= [-0.07904709, -0.04090177, 0.45242203, 0.88735163],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
        depth_camera5 = dict(
            pos =[4.93975558, -4.04511767, 2.0],  # Position the camera above the scene
            quat= [0.88589883, 0.45151363, 0.04865823, 0.09461013],  # Looking forward and downward at 45-degree angle
            camera_attribs=dict(fovy="100"),
        ),
    ),
}


def deep_update(d, u):
    """
    Copied from https://stackoverflow.com/a/3233356
    """
    import collections

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_robot_cam_configs(robot, layout_id = None):
    from copy import deepcopy

    default_configs = deepcopy(CAM_CONFIGS["DEFAULT"])
    if layout_id is not None:
        layout_cam_configs= deepcopy(CAM_CONFIGS_FOR_LAYOUT[layout_id])
        default_configs = deep_update(default_configs, layout_cam_configs)
    robot_specific_configs = deepcopy(CAM_CONFIGS.get(robot, {}))
    return deep_update(default_configs, robot_specific_configs)
