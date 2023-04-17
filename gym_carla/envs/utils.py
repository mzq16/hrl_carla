import pygame
import skimage
import numpy as np
from mmdet3d.datasets.pipelines import Compose
from PIL import Image
import torch


def init_render(width, height):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    return screen


def display_surface(image):
    # image.convert(cc.segment)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    return surface

def preprocess_img(image):
    '''
    (B,G,R,M) -> (R,G,B), I forget what M is 
    '''
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def preprocess_lidar(lidar):
    '''
    (x,y,z,I)
    '''
    
    points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points


def display_from_surface(screen, surface):
    if surface is not None:
        screen.blit(surface, (0, 0))


def action_to_control(action):
    acc = action[0]
    steer = action[1]
    steer = np.clip(steer,-1,1)
    if acc > 0:
        throttle = np.clip(acc, 0, 1)
        #throttle = acc
        brake = 0
    else:
        throttle = 0
        brake = np.clip(-acc, 0, 1)
        #brake = np.abs(acc)
    return throttle, brake, steer

def timestamp_tran(my_timestamp, timestamp):
    _timestamp = my_timestamp
    _timestamp['step'] = timestamp.frame-_timestamp['start_frame']
    _timestamp['frame'] = timestamp.frame
    _timestamp['wall_time'] = timestamp.platform_timestamp
    _timestamp['relative_wall_time'] = _timestamp['wall_time'] - _timestamp['start_wall_time']
    _timestamp['simulation_time'] = timestamp.elapsed_seconds
    _timestamp['relative_simulation_time'] = _timestamp['simulation_time'] \
        - _timestamp['start_simulation_time']
    return _timestamp

def proprecess_gps(gps_data):
    return [gps_data.latitude, gps_data.longitude, gps_data.altitude]

def proprecess_imu(imu_data):
    return [imu_data.accelerometer.x, imu_data.accelerometer.y, \
        imu_data.accelerometer.z, imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z, imu_data.compass]

def cast_angle(x):
    # cast angle to [-180, +180)
    return (x+180.0)%360.0-180.0


def create_compose():
    d1={'type': 'ImageAug3D', 'final_dim': [256, 704], 'resize_lim': [0.48, 0.48], 'bot_pct_lim': [0.0, 0.0], 'rot_lim': [0.0, 0.0], 'rand_flip': False, 'is_train': False}
    d2={'type': 'ImageNormalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    l =[d1,d2]
    compose = Compose(l)
    return compose

def get_bev_data(info, cam, lidar_np, my_data: dict, device):
    cam_in_list = []
    for i in range(6):
        tmp_in = np.eye(4)
        cam_name = f'camera_{i}'
        tmp_in[:3,:3] = info[cam_name]['in']
        cam_in_list.append(tmp_in)
    cam_in = np.array(cam_in_list)
    # cam_in.shape

    # 2 lidar2camera & camera2lidar
    # 2.1 rotation lidar2camera = lidar2world ->world2camera
    # 2.2 translation 
    w2l = info['lidar']['ex']
    l_left2right = np.array([[0,1,0],
                            [1,0,0],
                            [0,0,1]])
    c_left2right = np.array([[0,1,0],
                            [0,0,-1],
                            [1,0,0]])
    aug_l_left2right = np.eye(4)
    aug_l_left2right[:3,:3] = l_left2right
    aug_c_left2right = np.eye(4)
    aug_c_left2right[:3,:3] = c_left2right
    lidar2camera_list = []
    camera2lidar_list = []
    for i in range(6):
        cam_name = f'camera_{i}'
        w2c = info[cam_name]['ex'] 
        w2c_right = np.matmul(aug_c_left2right, w2c)
        w2l_right = np.matmul(aug_l_left2right, w2l)
        l_right2w = np.linalg.inv(w2l_right)
        l2c = np.matmul(w2c_right, l_right2w)
        c2l = np.linalg.inv(l2c)
        lidar2camera_list.append(l2c)
        camera2lidar_list.append(c2l)
    lidar2camera = np.array(lidar2camera_list)
    camera2lidar = np.array(camera2lidar_list)

    # 3 lidar2image: lidar2cam -> cam2img(intrinsic)
    lidar2image_list = []
    for i in range(6):
        tmp_l2c = lidar2camera[i]
        tmp_cam_in = cam_in[i]
        lidar2image_list.append(np.matmul(tmp_cam_in, tmp_l2c))
    lidar2image = np.array(lidar2image_list)

    data_img = {}
    img_list = []
    #CAM_NAME = ['camera_front', 'camera_front_right', 'camera_front_left',  'camera_back', 'camera_right', 'camera_left']
    for i in range(6):
        name = f'camera_{i}'
        img_list.append(Image.fromarray(cam[name]))
    #img = np.array(img_list)
    data_img['img'] = img_list
    data_img['ori_shape'] = cam[name].shape[1], cam[name].shape[0]
    compose = create_compose()
    data_ii = compose(data_img)
    img = np.stack(data_img['img'])
    aug_img = np.stack(data_img['img_aug_matrix'])
    img = torch.tensor(img).to(device=device)
    aug_img = torch.tensor(aug_img).to(device=device)

    # revise lidar 
    # (n,4)->(n,5), padding with 0
    lidar_len = lidar_np.shape[0]
    padding = np.zeros((lidar_len,1))
    lidar = np.hstack([lidar_np,padding])
    lidar = torch.tensor(lidar, dtype=torch.float32).to(device=device)
    my_data['points'][0] = lidar

    cam_in = torch.tensor(cam_in, dtype=torch.float32).to(device=device)
    my_data['camera_intrinsics'][0] = cam_in

    lidar2camera = torch.tensor(lidar2camera, dtype=torch.float32).to(device=device)
    my_data['lidar2camera'][0] = lidar2camera

    camera2lidar = torch.tensor(camera2lidar, dtype=torch.float32).to(device=device)
    my_data['camera2lidar'][0] = camera2lidar

    lidar2image = torch.tensor(lidar2image, dtype=torch.float32).to(device=device)
    my_data['lidar2image'][0] = lidar2image

    return my_data
