import matplotlib.pyplot as plt
import os
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.nuscenes_buw import NuScenesBUW
import numpy as np
import time
import pickle
import config as cfg
import random
from tqdm import tqdm

nusc = NuScenes(
    version='v1.0-trainval',
    dataroot=cfg.data_path,
    verbose=True)

nuscex = NuScenesExplorer(nusc)
nuscbuw = NuScenesBUW(nusc)

## chose the right GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= cfg.CUDA_VISIBLE_DEVICES

def get_settings(desc_list):
    '''
    get all information about the scenes like weather, daytime etc
    '''
    infos = []
    for i_info in range(len(desc_list)):
        infos.append(desc_list[i_info].lower() + ', ' + location_list[i_info].lower() + ', ' + 'annotations:' + str(
            ann_count_list[i_info]))  # add informations in list

    infos.sort()  # for scenes in nunerical order
    return infos


def preprocessing(location, condition, infos):
    '''
    seperate the nuScenes data in different locations and conditions
    locations: Boston and Singapore
    conditions: all, rain, night
    '''
    data = []
    location = location.lower()
    if location == 'boston':
        print('Location: Boston')
    elif location == 'singapore':
        print('Location: Singapore')
    else:
        print('Location: all Locations')

    if condition == 'rain':
        print('Condition: Rain')
    elif condition == 'night':
        print('Condition: Night')
    elif condition == 'all':
        print('Condition: all Conditions')
    else:
        print('Condition: Day and good weather')

    for i_prep in range(len(infos)):
        if location == 'all':
            if condition == 'all':
                data.append(infos[i_prep])  # get all conditions
            elif condition == 'day':
                if not 'rain' in infos[i_prep] and not 'night' in infos[i_prep]:
                    data.append(infos[i_prep])
            else:
                if condition in infos[i_prep]:
                    data.append(infos[i_prep])
        else:
            if condition == 'all':
                if infos[i_prep].split(', ')[-2].split('-')[0] == location:
                    data.append(infos[i_prep])
            elif condition == 'day':
                if infos[i_prep].split(', ')[-2].split('-')[0] == location and not 'rain' in infos[
                    i_prep] and not 'night' in infos[i_prep]:
                    data.append(infos[i_prep])
            else:
                if infos[i_prep].split(', ')[-2].split('-')[0] == location and str(condition) in infos[i_prep]:
                    data.append(infos[i_prep])

    print('Number of Scenes: ', len(data))

    # get scene numbers
    scene_nrs = []
    for i_nr in range(len(data)):
        scene_nrs.append(data[i_nr].split(',')[0].split('-')[1])

    return data, scene_nrs


def split_test_scenes_night(nr_night_scenes):
    '''
    first: write a list with night pics and write numbers of scenes for test data
    second: mix all scenes and delete the picked night pics for test data
    '''
    data, scene_nrs = preprocessing('All', 'night', infos)  # look in table for night data
    list_nrs = [scene_nrs[i] for i in range(len(scene_nrs))]

    new_order = []
    for i in range(len(list_nrs)):
        random_item_from_list = random.choice(list_nrs)
        list_nrs.remove(random_item_from_list)
        new_order.append(random_item_from_list)

    # ca. (25 scenes) of the night pics should be test data
    test_night_sc_nr = new_order[:nr_night_scenes]

    data, scene_nrs = preprocessing('All', 'all', infos)  # look in table for all data
    list_nrs = [scene_nrs[i] for i in range(len(scene_nrs))]

    new_order = []
    for i in range(len(list_nrs)):
        random_item_from_list = random.choice(list_nrs)
        list_nrs.remove(random_item_from_list)
        new_order.append(random_item_from_list)

    delete = test_night_sc_nr
    for i in range(len(delete)):
        new_order = list(filter(lambda num: num != delete[i], new_order))
    scene_nrs = new_order
    return scene_nrs, test_night_sc_nr


def seperate_data(scene_nrs, test_nrs):
    '''
    seperates all scene numbers into train, val and test data
    test data was seperated in the step before, because we want only night pics
    '''
    data_train = []
    data_val = []
    data_ratio = (0.9, 0.1)
    nr_train = int(np.round(data_ratio[0] * len(scene_nrs), 0))
    nr_val = int(np.round(data_ratio[1] * len(scene_nrs), 0))
    # nr_test = int(np.round(data_ratio[2] * len(scene_nrs),0))

    for i_sep in range(len(scene_nrs)):
        if i_sep <= nr_train:
            data_train.append(scene_nrs[i_sep])
        else:
            data_val.append(scene_nrs[i_sep])

    data_test = test_nrs  # includes the night data
    seperated_data = [data_train, data_val, data_test]
    return seperated_data


def get_all_scene_tokens_and_annotation():
    '''
    get all informations about each scene:
        scene nr
        pic nr
        pic token
        annotation tokens
        timestamp
        filename
    '''
    all_scenes = []
    dict_all_tokens_and_annotation = {}
    for i_scene in tqdm(range(len(nusc.scene))):
        scene_number_infos = nusc.scene[i_scene]
        first_token_of_scene = scene_number_infos['first_sample_token']
        number_of_pics_in_scene = scene_number_infos['nbr_samples']
        scene_num = nusc.scene[i_scene]['name'].split('-')[1]

        actual_token = first_token_of_scene
        for i_tokens in range(number_of_pics_in_scene):
            scene_infos = nusc.get('sample', actual_token)

            if i_tokens == 0:
                dict_all_tokens_and_annotation['scene'] = scene_num
                dict_all_tokens_and_annotation['pic_number'] = str(i_tokens).zfill(2)
                dict_all_tokens_and_annotation['pic_token'] = actual_token
                dict_all_tokens_and_annotation['tokens_for_annotations'] = nusc.get('sample', actual_token)['anns']
                dict_all_tokens_and_annotation['timestamp'] = str(nusc.get('sample', actual_token)['timestamp'])
                dict_all_tokens_and_annotation['filename'] = \
                    str(nusc.get('sample_data', nusc.get('sample', actual_token)['data']['CAM_FRONT'])['filename'])
                all_scenes.append(dict_all_tokens_and_annotation)
                actual_token = scene_infos['next']
                dict_all_tokens_and_annotation = {}
            else:
                dict_all_tokens_and_annotation['scene'] = scene_num
                dict_all_tokens_and_annotation['pic_number'] = str(i_tokens).zfill(2)
                dict_all_tokens_and_annotation['pic_token'] = actual_token
                dict_all_tokens_and_annotation['tokens_for_annotations'] = nusc.get('sample', actual_token)['anns']
                dict_all_tokens_and_annotation['timestamp'] = str(nusc.get('sample', actual_token)['timestamp'])
                dict_all_tokens_and_annotation['filename'] = \
                    str(nusc.get('sample_data', nusc.get('sample', actual_token)['data']['CAM_FRONT'])['filename'])
                all_scenes.append(dict_all_tokens_and_annotation)
                actual_token = scene_infos['next']
                dict_all_tokens_and_annotation = {}
    return all_scenes


def get_all_sensor_tokens(all_scenes, seperated_data, location, condition, path):
    '''
    get a list of all important tokens:
        scene nr
        pic nr
        timestamp
        filename
        pic token
        camera token
        annotation tokens
    '''
    scenes_train = seperated_data[0]
    scenes_val = seperated_data[1]
    scenes_test = seperated_data[2]

    sensor_token_train = []
    sensor_token_test = []
    sensor_token_val = []
    dict_sensors_train = {}
    dict_sensors_test = {}
    dict_sensors_val = {}
    pointsensor_channel = 'RADAR_FRONT'
    i_test = 0
    i_train = 0
    i_val = 0
    #TODO: delete the two loops
    for scene in scenes_train:
        for i in range(len(all_scenes)):
            if scene == all_scenes[i]['scene']:
                dict_sensors_train['scene'] = all_scenes[i]['scene']
                dict_sensors_train['pic_number'] = all_scenes[i]['pic_number']
                dict_sensors_train['pic_token'] = all_scenes[i]['pic_token']
                dict_sensors_train['timestamp'] = all_scenes[i]['timestamp']
                dict_sensors_train['filename'] = all_scenes[i]['filename']
                dict_sensors_train['pointsensor_token'], dict_sensors_train['camera_token'], dict_sensors_train[
                    'scene_token'] = nuscbuw.get_tokens(
                    all_scenes[i]['pic_token'],
                    pointsensor_channel=pointsensor_channel)
                dict_sensors_train['annotation_tokens'] = all_scenes[i]['tokens_for_annotations']
                sensor_token_train.append(dict_sensors_train)
                dict_sensors_train = {}
    for scene in scenes_val:
        for i in range(len(all_scenes)):
            if scene == all_scenes[i]['scene']:
                dict_sensors_val['scene'] = all_scenes[i]['scene']
                dict_sensors_val['pic_number'] = all_scenes[i]['pic_number']
                dict_sensors_val['pic_token'] = all_scenes[i]['pic_token']
                dict_sensors_val['timestamp'] = all_scenes[i]['timestamp']
                dict_sensors_val['filename'] = all_scenes[i]['filename']
                dict_sensors_val['pointsensor_token'], dict_sensors_val['camera_token'], dict_sensors_val[
                    'scene_token'] = nuscbuw.get_tokens(
                    all_scenes[i]['pic_token'],
                    pointsensor_channel=pointsensor_channel)
                dict_sensors_val['annotation_tokens'] = all_scenes[i]['tokens_for_annotations']
                sensor_token_val.append(dict_sensors_val)
                dict_sensors_val = {}
    for scene in scenes_test:
        for i in range(len(all_scenes)):
            if scene == all_scenes[i]['scene']:
                dict_sensors_test['scene'] = all_scenes[i]['scene']
                dict_sensors_test['pic_number'] = all_scenes[i]['pic_number']
                dict_sensors_test['pic_token'] = all_scenes[i]['pic_token']
                dict_sensors_test['timestamp'] = all_scenes[i]['timestamp']
                dict_sensors_test['filename'] = all_scenes[i]['filename']
                dict_sensors_test['pointsensor_token'], dict_sensors_test['camera_token'], dict_sensors_test[
                    'scene_token'] = nuscbuw.get_tokens(
                    all_scenes[i]['pic_token'],
                    pointsensor_channel=pointsensor_channel)
                dict_sensors_test['annotation_tokens'] = all_scenes[i]['tokens_for_annotations']
                sensor_token_test.append(dict_sensors_test)
                dict_sensors_test = {}

    # save the tokens into pickle-file
    if not os.path.exists(path + "/Tokens"):
        os.makedirs(path + "/Tokens")

    save_sensor_token_train = open(path + "/Tokens/sensor_token_train_{}_{}.pkl".format(location, condition), "wb")
    pickle.dump(sensor_token_train, save_sensor_token_train)
    save_sensor_token_train.close()

    save_sensor_token_test = open(path + "/Tokens/sensor_token_test_{}_{}.pkl".format(location, condition), "wb")
    pickle.dump(sensor_token_test, save_sensor_token_test)
    save_sensor_token_test.close()

    save_sensor_token_val = open(path + "/Tokens/sensor_token_val_{}_{}.pkl".format(location, condition), "wb")
    pickle.dump(sensor_token_val, save_sensor_token_val)
    save_sensor_token_val.close()

    return sensor_token_train, sensor_token_test, sensor_token_val


def load_all_tokens(location, condition, path):
    sensor_token_train = pickle.load(
        open(path + '/Tokens/sensor_token_train_{}_{}.pkl'.format(location, condition), "rb"))

    sensor_token_val = pickle.load(open(path + '/Tokens/sensor_token_val_{}_{}.pkl'.format(location, condition), "rb"))
    sensor_token_test = pickle.load(
        open(path + '/Tokens/sensor_token_test_{}_{}.pkl'.format(location, condition), "rb"))

    print('Tokens successfully loaded')
    return sensor_token_train, sensor_token_val, sensor_token_test


def data_prep(scene_package, pref_class, use_sweeps):
    plot_pics = [False, False, False]  # if True: show pictures, the first entry is the actual picture
    nr_scenes = list_scenes(scene_package)  # contains actual package (train or test)
    list_tokens_scene = []
    nr_row_matrix = 0  # iteraring variable for x_train

    slices = cfg.slices
    channels = 4  # number of informations for each radar point
    time_steps = 3
    time_sweeps = 3
    train_size = (len(scene_package) - (time_steps - 1) * len(nr_scenes) + 1)  # without 2 pictures for every scene
    matrix_x_train = np.zeros((train_size, slices, time_steps, channels))
    matrix_y_train = np.zeros((train_size, slices, 1, 1))

    list_tokens_by_scene = sum_tokens_by_scene(scene_package)
    print('*** start of data preparation for training ***')
    for i_scene in tqdm(range(len(list_tokens_by_scene))):  # run through every scene

        # run through every pic in scene
        annotation = []
        for i_pic in range(2, len(list_tokens_by_scene[i_scene]['pointsensor_tokens'])):
            annotation = get_annotation_name(i_scene, i_pic, list_tokens_by_scene, time_steps)

            if use_sweeps:
                matrix_pic_data = get_pic_data_sweep(list_tokens_by_scene, time_sweeps, i_scene, i_pic,
                                                     annotation, plot_pics, slices, pref_class)
            else:
                matrix_pic_data, _, _ = get_pic_data(list_tokens_by_scene, time_steps, i_scene, i_pic,
                                                     annotation, plot_pics, slices, pref_class)

            ### write informations in matrix_x/y_train
            matrix_x_train[nr_row_matrix, :, 0, :] = matrix_pic_data[0][2:6].T  # actual pic
            matrix_x_train[nr_row_matrix, :, 1, :] = matrix_pic_data[1][2:6].T
            matrix_x_train[nr_row_matrix, :, 2, :] = matrix_pic_data[2][2:6].T
            matrix_y_train[nr_row_matrix, :, 0, 0] = matrix_pic_data[0][11]  # groundtruth
            nr_row_matrix += 1

    return matrix_x_train, matrix_y_train

def list_scenes(scene_package): # insert train or test-package
    # list all scene numbers
    nr_scenes = []
    for i in range(len(scene_package)):
        nr_scenes.append(scene_package[i]['scene'])
    nr_scenes = list(dict.fromkeys(nr_scenes))
    return nr_scenes

def sum_tokens_by_scene(scene_package):
    sum_tokens = {}
    list_tokens_by_scene = []
    runner_scene = 0  # variable, which contains actual scene

    sum_tokens['scene'] = []
    sum_tokens['pic_tokens'] = []
    sum_tokens['timestamp'] = []
    sum_tokens['filename'] = []
    sum_tokens['pointsensor_tokens'] = []
    sum_tokens['camera_tokens'] = []
    sum_tokens['scene_token'] = []
    sum_tokens['annotation_list_tokens'] = []

    for i_scene in range(len(scene_package)):
        if i_scene < 1:  # get first actual scene
            runner_scene = scene_package[i_scene]['scene']
            sum_tokens['scene'] = runner_scene
            sum_tokens['pic_tokens'].append(scene_package[i_scene]['pic_token'])
            sum_tokens['timestamp'].append(scene_package[i_scene]['timestamp'])
            sum_tokens['filename'].append(scene_package[i_scene]['filename'])
            sum_tokens['pointsensor_tokens'].append(scene_package[i_scene]['pointsensor_token'])
            sum_tokens['camera_tokens'].append(scene_package[i_scene]['camera_token'])
            sum_tokens['scene_token'].append(scene_package[i_scene]['scene_token'])
            sum_tokens['annotation_list_tokens'].append(scene_package[i_scene]['annotation_tokens'])
        elif scene_package[i_scene]['scene'] == str(runner_scene):
            sum_tokens['pic_tokens'].append(scene_package[i_scene]['pic_token'])
            sum_tokens['timestamp'].append(scene_package[i_scene]['timestamp'])
            sum_tokens['filename'].append(scene_package[i_scene]['filename'])
            sum_tokens['pointsensor_tokens'].append(scene_package[i_scene]['pointsensor_token'])
            sum_tokens['camera_tokens'].append(scene_package[i_scene]['camera_token'])
            sum_tokens['annotation_list_tokens'].append(scene_package[i_scene]['annotation_tokens'])
        else:
            list_tokens_by_scene.append(sum_tokens)
            runner_scene = str(scene_package[i_scene]['scene'])  # get actual scene
            sum_tokens = {}
            sum_tokens['scene'] = runner_scene
            sum_tokens['pic_tokens'] = []
            sum_tokens['timestamp'] = []
            sum_tokens['filename'] = []
            sum_tokens['pointsensor_tokens'] = []
            sum_tokens['camera_tokens'] = []
            sum_tokens['annotation_list_tokens'] = []

            sum_tokens['pic_tokens'].append(scene_package[i_scene]['pic_token'])
            sum_tokens['timestamp'].append(scene_package[i_scene]['timestamp'])
            sum_tokens['filename'].append(scene_package[i_scene]['filename'])
            sum_tokens['pointsensor_tokens'].append(scene_package[i_scene]['pointsensor_token'])
            sum_tokens['camera_tokens'].append(scene_package[i_scene]['camera_token'])
            sum_tokens['scene_token'] = scene_package[i_scene]['scene_token']
            sum_tokens['annotation_list_tokens'].append(scene_package[i_scene]['annotation_tokens'])

    # add last scene
    sum_tokens['pic_tokens'].append(scene_package[i_scene]['pic_token'])
    sum_tokens['timestamp'].append(scene_package[i_scene]['timestamp'])
    sum_tokens['filename'].append(scene_package[i_scene]['filename'])
    sum_tokens['pointsensor_tokens'].append(scene_package[i_scene]['pointsensor_token'])
    sum_tokens['camera_tokens'].append(scene_package[i_scene]['camera_token'])
    sum_tokens['annotation_list_tokens'].append(scene_package[i_scene]['annotation_tokens'])
    list_tokens_by_scene.append(sum_tokens)

    return list_tokens_by_scene


def get_annotation_name(nr_scene, nr_pic, list_tokens_by_scene, time_steps):
    annotation = []
    annotation_step = []
    for i_step in range(0, time_steps):
        for i_ann in range(len(list_tokens_by_scene[nr_scene]['annotation_list_tokens'][nr_pic - i_step])):
            my_annotation_metadata = nusc.get('sample_annotation',
                                              list_tokens_by_scene[nr_scene]['annotation_list_tokens'][nr_pic - i_step][
                                                  i_ann])
            category_name = my_annotation_metadata['category_name']
            category_name = category_name.split('.')
            annotation_step.append(category_name[0])
        annotation.append(annotation_step)
        annotation_step = []
    return annotation


def get_pic_data(
        list_tokens_by_scene,
        time_steps,
        i_scene,
        i_pic,
        annotation,
        plot_pics,
        slices,
        pref_class):
    ### get rectangles and points for the step_size
    points = []  # for radar points
    coloring = []  # depth information
    points_all = []  # for radar points
    im = []  # picture
    matrix_pic_data = []
    data_path = []
    rectangles = []

    for i_step in range(0, time_steps):
        data_path_step = list_tokens_by_scene[i_scene]['filename'][i_pic - i_step].split('/')[2]
        data_path.append(data_path_step)

        rectangle_step, _, _ = nuscbuw.render_annotation_cam_front(
            list_tokens_by_scene[i_scene]['annotation_list_tokens'][i_pic - i_step],  # list of annotations
            annotation[i_step],
            pref_class=pref_class,
            plot_pics=plot_pics[i_step])  # first entry is the actual picture
        rectangles.append(rectangle_step)

        points_step, coloring_step, im_step, points_all_step = nuscbuw.map_pointcloud_to_image_buw(
            pointsensor_token=list_tokens_by_scene[i_scene]['pointsensor_tokens'][i_pic - i_step],
            camera_token=list_tokens_by_scene[i_scene]['camera_tokens'][i_pic - i_step],
            min_dist=1.0,
            render_intensity=False)

        points.append(points_step)
        coloring.append(coloring_step)
        points_all.append(points_all_step)
        im.append(im_step)  # just first entry important

        ### write results of the pictures in matrix
        matrix_pic_data_step, _ = nuscbuw.get_matrix_radar(
            rectangle_step,
            im_step,  # just for size
            points_step,
            coloring_step,
            slices,
            points_all_step)

        matrix_pic_data.append(matrix_pic_data_step)

        # scatter just existing points and delete zero columns
        mask = np.ones(slices, dtype=bool)
        mask[matrix_pic_data[0][1, :] == 0] = False

        matrix_scatter_x = matrix_pic_data[0][1, :][mask]
        matrix_scatter_y = matrix_pic_data[0][2, :][mask]
        matrix_scatter_z = matrix_pic_data[0][3, :][mask]

        # for pic
        if plot_pics[i_step] == True:
            plt.scatter(matrix_scatter_x, matrix_scatter_y, c=matrix_scatter_z, s=50, zorder=2)
            # plt.savefig('PATH/pic_number_{}.png'.format(i_run))

    return matrix_pic_data, data_path, rectangles


def get_pic_data_sweep(list_tokens_by_scene,
                       time_sweeps,  # number of sweeps to use (included current pic)
                       i_scene,
                       i_pic,
                       annotation,
                       plot_pics,
                       slices,
                       pref_class):
    path_sweeps = '/home/datasets/nuScenes/sweeps/RADAR_FRONT/'
    sweeps = pickle.load(open('INSERT_PATH', "rb"))
    for i_sweep in range(len(sweeps)):  # get current sweeps
        if int(sweeps[i_sweep]['scene']) == i_scene + 1 and int(sweeps[i_sweep]['pic_number']) == i_pic:
            break
    current_sweeps = sweeps[i_sweep]
    matrix_pic_data = []

    # added for sweep
    data_path = list_tokens_by_scene[i_scene]['filename'][i_pic].split('/')[2]

    rectangle_step, _ = nuscbuw.render_annotation_cam_front(
        list_tokens_by_scene[i_scene]['annotation_list_tokens'][i_pic],  # list of annotations
        annotation[0],
        pref_class=pref_class,
        plot_pics=plot_pics[0])  # first entry is the actual picture

    sweep_points, sweep_coloring, im, sweep_points_all = nuscbuw.map_pointcloud_sweep(
        pointsensor_token=list_tokens_by_scene[i_scene]['pointsensor_tokens'][i_pic],
        camera_token=list_tokens_by_scene[i_scene]['camera_tokens'][i_pic],
        min_dist=1.0,
        current_sweeps=current_sweeps,
        scene=i_scene,
        pic=i_pic,
        time_sweeps=time_sweeps,
        path_sweeps=path_sweeps,
        render_intensity=False)

    ### write results of the pictures in matrix
    for i_sw in range(0, time_sweeps):
        matrix_pic_data_step, _ = nuscbuw.get_matrix_radar(
            rectangle_step,
            im,  # just for size
            sweep_points[i_sw],
            sweep_coloring[i_sw],
            slices,
            sweep_points_all[i_sw])

        matrix_pic_data.append(matrix_pic_data_step)

    # for pic
    if plot_pics[0] == True:
        plt.scatter(matrix_scatter_x, matrix_scatter_y, c='white', s=50, zorder=3)
        plt.show()

    return matrix_pic_data, data_path, rectangle_step


def get_input_matrix (scene_package, pref_class, name_affix, package_name, path, use_sweeps):
    plot_pics = [False, False, False]  # if True: show pictures, the first entry is the actual picture
    slices = cfg.slices
    time_steps = 1
    time_sweeps = 3
    rectangle = []
    list_input = []
    dict_input = {}

    list_tokens_by_scene = sum_tokens_by_scene(scene_package)
    print('*** start for saving the input matrix ***')
    for i_scene in tqdm(range(len(list_tokens_by_scene))):  # run through every scene
        current_state = round((100 / len(list_tokens_by_scene)) * i_scene, 2)
        # print('Szene: ', i_scene, ' ', current_state, '%')

        for i_pic in range(len(list_tokens_by_scene[i_scene]['pointsensor_tokens'])):
            annotation = get_annotation_name(i_scene, i_pic, list_tokens_by_scene, time_steps)

            if use_sweeps:
                matrix_pic_data, data_path, rectangles = get_pic_data_sweep(list_tokens_by_scene, time_sweeps, i_scene,
                                                                    i_pic, annotation, plot_pics, slices, pref_class)
            else:
                matrix_pic_data, data_path, rectangles = get_pic_data(list_tokens_by_scene, time_steps, i_scene, i_pic,
                                                                      annotation, plot_pics, slices, pref_class)

            ### write input file as dictionary
            dict_input['scene'] = list_tokens_by_scene[i_scene]['scene']
            dict_input['pic'] = i_pic
            dict_input['file_path'] = list_tokens_by_scene[i_scene]['filename'][i_pic]
            dict_input['radar_data'] = matrix_pic_data[0][2:6].T
            dict_input['bbox'] = rectangles
            dict_input['gt'] = matrix_pic_data[0][11]
            list_input.append(dict_input)
            dict_input = {}

    save_input = open(path + f'/data_{cfg.slices}/input_yodar_{package_name}_{name_affix[0]}_{name_affix[1]}.pkl', "wb")
    pickle.dump(list_input, save_input)
    save_input.close


def prep_kitti(list_tokens_by_scene,
               time_steps,
               pref_class,
               affix,
               path,
               slices,
               affix_scene):

    if not os.path.exists(path + f'/KITTI_format_{slices}'):
        os.makedirs(path + f'/KITTI_format_{slices}')
    file_name = path +  f'/KITTI_format_{slices}/nusc_{affix_scene}.txt'
    
    # delete file if existent
    if os.path.exists(file_name):
        os.remove(file_name)
    
    ##### write txt-file for kitti #####
    ##### scheme: pic_path xmin, ymin, xmax, ymax, category_class
    anntoken = []
    dict_anntoken = {}
    print('*** start for saving the KITTI format ***')
    for i_scene in tqdm(range(len(list_tokens_by_scene))):
        current_state = round((100 / len(list_tokens_by_scene)) * i_scene, 2)

        for i_pic in range(len(list_tokens_by_scene[i_scene]['pointsensor_tokens'])):
            annotation = get_annotation_name(i_scene, i_pic, list_tokens_by_scene, time_steps)
            data_path = list_tokens_by_scene[i_scene]['filename'][i_pic]
            data_path = cfg.data_path + data_path #path from dataset

            # scheme rectangle: (xmax, xmin, ymax, ymin)
            rectangle, _, list_anntoken = nuscbuw.render_annotation_cam_front(
                list_tokens_by_scene[i_scene]['annotation_list_tokens'][i_pic],  # list of annotations
                annotation[0],  # always first time_step entry
                pref_class=pref_class,
                plot_pics=False)  # first entry is the actual picture

            if pref_class == 'vehicle':
                category_class = '000000'
            else:
                category_class = '000001'

            with open(file_name, "a") as text_file:  # a for append, w for write
                print(data_path, end=' ', file=text_file) # write image path 
                for i_rect in range(len(rectangle)):
                    print(str(int(round(rectangle[i_rect][0]))).zfill(6) + ',',
                          str(int(round(rectangle[i_rect][2]))).zfill(6) + ',',
                          str(int(round(rectangle[i_rect][1]))).zfill(6) + ',',
                          str(int(round(rectangle[i_rect][3]))).zfill(6) + ',',
                          category_class,
                          end=' ',
                          file=text_file)
                print(end='\n', file=text_file)


            ### scheme for distance-calculation:
            #dict_anntoken['scene'] = i_scene
            dict_anntoken['scene'] = list_tokens_by_scene[i_scene]['scene']
            dict_anntoken['pic_num'] = i_pic
            dict_anntoken['path'] = data_path
            tmp_anntoken    = []
            tmp_dist        = []
            tmp_rec         = []
            for i_ann in range(len(list_anntoken)):
                tmp_anntoken.append(list_anntoken[i_ann]['anntoken'])
                tmp_dist.append(str(list_anntoken[i_ann]['distance']))
                rec = [int(round(rectangle[i_ann][1])), #xmin
                           int(round(rectangle[i_ann][3])), #ymin
                           int(round(rectangle[i_ann][0])), #xmax
                           int(round(rectangle[i_ann][2]))] #ymax
                tmp_rec.append(rec)
            dict_anntoken['anntoken'] = tmp_anntoken
            dict_anntoken['distance'] = tmp_dist
            dict_anntoken['rectangle'] =tmp_rec
            anntoken.append(dict_anntoken)
            dict_anntoken = {}

    #save pickle-file
    save_anntoken = open(path + f"/data_{slices}/anntoken_{affix_scene}.pkl", "wb")
    pickle.dump(anntoken, save_anntoken)
    save_anntoken.close()


if __name__ == '__main__':
    start = time.time()

    ### get settings from config.py
    path = cfg.path
    name_affix = cfg.conditions_night
    pref_class = cfg.pref_class
    use_sweeps = cfg.use_sweeps

    if cfg.save_tokens:
        ### get infos from each scene/image
        desc_list, desc_list_compact, length_time_list, location_list, ann_count_list = nuscbuw.list_scenes_buw() #for Informations about the scenes
        infos = get_settings(desc_list)

        ### seperate night scenes
        scene_nrs, test_nrs = split_test_scenes_night(cfg.nr_night_scenes)
        seperated_data =seperate_data(scene_nrs, test_nrs)

        all_scenes = get_all_scene_tokens_and_annotation()
        sensor_token_train, sensor_token_test, sensor_token_val = get_all_sensor_tokens(
                                    all_scenes, seperated_data, name_affix[0], name_affix[1], path)

    if cfg.load_tokens:
        ### get tokens
        sensor_token_train, sensor_token_val, sensor_token_test = load_all_tokens(name_affix[0], name_affix[1], path)

    if cfg.save_training_data:
        print('start saving training data')
        ### get infos from each scene/image
        desc_list, desc_list_compact, length_time_list, location_list, ann_count_list = nuscbuw.list_scenes_buw() #for Informations about the scenes
        infos = get_settings(desc_list)

        ### check if folder existant
        if not os.path.exists(path + f"/data_{cfg.slices}"):
            os.makedirs(path + f"/data_{cfg.slices}")

        ### save data as train, test and val
        matrix_x, matrix_y = data_prep(sensor_token_train, pref_class, use_sweeps)
        np.save(path + f"/data_{cfg.slices}/x_train_{name_affix[0]}_{name_affix[1]}.npy", matrix_x)
        np.save(path + f"/data_{cfg.slices}/y_train_{name_affix[0]}_{name_affix[1]}.npy", matrix_y)

        matrix_x, matrix_y = data_prep(sensor_token_val, pref_class, use_sweeps)
        np.save(path + f"/data_{cfg.slices}/x_val_{name_affix[0]}_{name_affix[1]}.npy", matrix_x)
        np.save(path + f"/data_{cfg.slices}/y_val_{name_affix[0]}_{name_affix[1]}.npy", matrix_y)

        matrix_x, matrix_y = data_prep(sensor_token_test, pref_class, use_sweeps)
        np.save(path + f"/data_{cfg.slices}/x_test_{name_affix[0]}_{name_affix[1]}.npy", matrix_x)
        np.save(path + f"/data_{cfg.slices}/y_test_{name_affix[0]}_{name_affix[1]}.npy", matrix_y)
        print('save training data done')

    ### for input matrix
    if cfg.get_input_matrix:
        print('start saving input matrix')
        name_affix = cfg.name_affix
        pref_class = cfg.pref_class
        use_sweeps = cfg.use_sweeps
        sensor_token_train, sensor_token_val, sensor_token_test = load_all_tokens(name_affix[0], name_affix[1], path)
        get_input_matrix(sensor_token_train, pref_class, name_affix, 'train', path, use_sweeps)
        get_input_matrix(sensor_token_val, pref_class, name_affix, 'val', path, use_sweeps)
        get_input_matrix(sensor_token_test, pref_class, name_affix, 'test', path, use_sweeps)
        print('save input matrix done')

    if cfg.get_kitti:
        print('start saving KITTI format')
        name_affix = cfg.name_affix
        pref_class = cfg.pref_class
        sensor_token_train, sensor_token_val, sensor_token_test = load_all_tokens(
            name_affix[0], name_affix[1], path)
        list_tokens_by_scene = sum_tokens_by_scene(sensor_token_test)
        time_steps = 1
        prep_kitti(list_tokens_by_scene, time_steps, pref_class, name_affix, path, cfg.slices, affix_scene='test')
        list_tokens_by_scene = sum_tokens_by_scene(sensor_token_val)
        prep_kitti(list_tokens_by_scene, time_steps, pref_class, name_affix, path, cfg.slices, affix_scene='val')
        list_tokens_by_scene = sum_tokens_by_scene(sensor_token_train)
        prep_kitti(list_tokens_by_scene, time_steps, pref_class, name_affix, path, cfg.slices, affix_scene='train')

        # get the first 10.000 frames for YOLO training
        with open(f'./KITTI_format_{cfg.slices}/nusc_train.txt') as file:
            all_lines = file.readlines()
        with open(f'./KITTI_format_{cfg.slices}/nusc_train_10000.txt', "a") as file:
            for i in range(10000):
                file.write(all_lines[i])
        print('save KITTI format done')

    end = time.time()
    print(f'time: {round((end - start), 2)} seconds = {round(((end - start) / 60), 2)} minutes')